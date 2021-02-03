import os, argparse, json, copy, time
from tqdm import tqdm
from functools import partial
import torch, torchvision
import numpy as np

import data, models 
import experiment_manager as xpm
from fl_devices import Client, Server


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

np.set_printoptions(precision=4, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--schedule", default="main", type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=None, type=int)
parser.add_argument("--reverse_order", default=False, type=bool)
parser.add_argument("--hp", default=None, type=str)

parser.add_argument("--DATA_PATH", default=None, type=str)
parser.add_argument("--RESULTS_PATH", default=None, type=str)
parser.add_argument("--CHECKPOINT_PATH", default=None, type=str)

args = parser.parse_args()



def run_experiment(xp, xp_count, n_experiments):

  t0 = time.time()
  print(xp)
  hp = xp.hyperparameters

  num_classes = {"cifar10" : 10, "cifar100" : 100}[hp["dataset"]]

  model_names = [model_name for model_name, k in hp["models"].items() for _ in range(k)]
  
  optimizer, optimizer_hp = getattr(torch.optim, hp["local_optimizer"][0]), hp["local_optimizer"][1]
  optimizer_fn = lambda x : optimizer(x, **{k : hp[k] if k in hp else v for k, v in optimizer_hp.items()})

  distill_optimizer, distill_optimizer_hp = getattr(torch.optim, hp["distill_optimizer"][0]), hp["distill_optimizer"][1]
  distill_optimizer_fn = lambda x : distill_optimizer(x, **{k : hp[k] if k in hp else v for k, v in distill_optimizer_hp.items()})


  train_data, test_data = data.get_data(hp["dataset"], args.DATA_PATH)
  all_distill_data = data.get_data(hp["distill_dataset"], args.DATA_PATH)

  np.random.seed(hp["random_seed"])
  torch.manual_seed(hp["random_seed"])

  n_distill = int(hp["n_distill_frac"] * len(all_distill_data))

  distill_data = data.IdxSubset(all_distill_data, np.random.permutation(len(all_distill_data))[:n_distill], return_index=True)
  public_data = data.IdxSubset(all_distill_data, np.random.permutation(len(all_distill_data))[n_distill:len(all_distill_data)], return_index=False)

  print(len(distill_data), len(public_data))

  client_loaders, test_loader = data.get_loaders(train_data, test_data, n_clients=len(model_names), 
        alpha=hp["alpha"], batch_size=hp["batch_size"], n_data=None, num_workers=0, seed=hp["random_seed"])
  distill_loader = torch.utils.data.DataLoader(distill_data, batch_size=hp["distill_batch_size"], shuffle=True, num_workers=8)
  public_loader = torch.utils.data.DataLoader(public_data, batch_size=128, shuffle=True, num_workers=8)

  clients = [Client(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes) for i, (loader, model_name) in enumerate(zip(client_loaders, model_names))]
  server = Server(np.unique(model_names), distill_optimizer_fn, test_loader, distill_loader, num_classes=num_classes)

  for client in clients:
    client.public_loader = public_loader
    client.distill_loader = distill_loader
    
  # print model
  models.print_model(clients[0].model)


  if "P" in hp["aggregation_mode"] or hp["aggregation_mode"] == "FedAUX":
    for model_name, model in server.model_dict.items():
      pretrained = hp["pretrained"] if hp["pretrained"] else "{}_{}.pth".format(model_name, hp["distill_dataset"])

      loaded_state = torch.load(args.CHECKPOINT_PATH + pretrained, map_location='cpu')
      loaded_layers = [key for key in loaded_state if key in model.state_dict()]
      model.load_state_dict(loaded_state, strict=False)      

    for client in clients:
      client.synchronize_with_server(server)

    print("Successfully loaded layers {} from".format(loaded_layers), pretrained)


    if hp["aggregation_mode"] == "FedAUX":
      print("Computing Scores...")

      for client in clients:
        client.scores = client.extract_features_and_compute_scores(client.loader, public_loader, distill_loader, lambda_reg=hp["lambda_reg_score"], eps_delt=hp["eps_delt"]) 
        
        if hp["save_scores"]:
          xp.log({"client_{}_scores".format(client.id) : client.scores.detach().cpu().numpy()})
      

  if "L" in hp["aggregation_mode"]:
    for client in clients:
      for name, param in client.model.named_parameters():
        if "classification_layer" not in name:
          param.requires_grad = False

    for model in server.model_dict.values():
      for name, param in model.named_parameters():
        if "classification_layer" not in name:
          param.requires_grad = False



  # Start Distributed Training Process
  print("Start Distributed Training..\n")
  t1 = time.time()
  xp.log({"prep_time" : t1-t0})

  xp.log({"server_val_{}".format(key) : value for key, value in server.evaluate_ensemble().items()})
  for c_round in range(1, hp["communication_rounds"]+1):

    participating_clients = server.select_clients(clients, hp["participation_rate"])
    xp.log({"participating_clients" : np.array([c.id for c in participating_clients])})

    for client in participating_clients:
      client.synchronize_with_server(server)

      train_stats = client.compute_weight_update(hp["local_epochs"], lambda_fedprox=hp["lambda_fedprox"] if "PROX" in hp["aggregation_mode"] else 0.0) 

      print(train_stats)


    # Averaging
    server.aggregate_weight_updates(participating_clients)
    avg_stats = server.evaluate_ensemble()

    xp.log({"averaging_{}".format(key) : value for key, value in avg_stats.items()})


    if hp["aggregation_mode"] in ["FedDF", "FedAUX", "FedDF+P"]:
      distill_mode = "weighted_logits_precomputed" if hp["aggregation_mode"]=="FedAUX" else "mean_logits"

      distill_stats = server.distill(participating_clients, hp["distill_epochs"], mode=distill_mode, num_classes=num_classes)
      xp.log({"distill_{}".format(key) : value for key, value in distill_stats.items()})


    # Logging
    if xp.is_log_round(c_round):
      print("Experiment: {} ({}/{})".format(args.schedule, xp_count+1, n_experiments))   
      
      xp.log({'communication_round' : c_round, 'epochs' : c_round*hp['local_epochs']})
      xp.log({key : clients[0].optimizer.__dict__['param_groups'][0][key] for key in optimizer_hp})
      
      # Evaluate  
      xp.log({"server_val_{}".format(key) : value for key, value in server.evaluate_ensemble().items()})

      xp.log({"epoch_time" : (time.time()-t1)/c_round})
      # Save results to Disk
      try:
        xp.save_to_disc(path=args.RESULTS_PATH, name=hp['log_path'])
      except:
        print("Saving results Failed!")

      # Timing
      e = int((time.time()-t1)/c_round*(hp['communication_rounds']-c_round))
      print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60), 
                "[{:.2f}%]\n".format(c_round/hp['communication_rounds']*100))

  # Save model to disk
  server.save_model(path=args.CHECKPOINT_PATH, name=hp["save_model"])
    
  # Delete objects to free up GPU memory
  del server; clients.clear()
  torch.cuda.empty_cache()


def run():


  experiments_raw = json.loads(args.hp)


  hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(x)][args.start:args.end]
  if args.reverse_order:
    hp_dicts = hp_dicts[::-1]
  experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]

  print("Running {} Experiments..\n".format(len(experiments)))
  for xp_count, experiment in enumerate(experiments):
    run_experiment(experiment, xp_count, len(experiments))


if __name__ == "__main__":
  run()
    