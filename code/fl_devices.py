import random
from tqdm import tqdm
from functools import partial

import torch
import torch.optim as optim
#from torchcontrib.optim import SWA
import torch.nn as nn
import numpy as np 

import models as model_utils
from sklearn.linear_model import LogisticRegression




device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

class Device(object):
  def __init__(self, loader):
    
    self.loader = loader

  def evaluate(self, loader=None):
    return eval_op(self.model, self.loader if not loader else loader)

  def save_model(self, path=None, name=None, verbose=True):
    if name:
      torch.save(self.model.state_dict(), path+name)
      if verbose: print("Saved model to", path+name)

  def load_model(self, path=None, name=None, verbose=True):
    if name:
      self.model.load_state_dict(torch.load(path+name))
      if verbose: print("Loaded model from", path+name)


    

class Client(Device):
  def __init__(self, model_name, optimizer_fn, loader, idnum=0, num_classes=10):
    super().__init__(loader)
    self.id = idnum

    self.model_name = model_name
    self.model_fn = partial(model_utils.get_model(self.model_name)[0], num_classes=num_classes)
    self.model = self.model_fn().to(device)

    self.W = {key : value for key, value in self.model.named_parameters()}

    self.optimizer_fn = optimizer_fn
    self.optimizer = self.optimizer_fn(self.model.parameters())   

    
  def synchronize_with_server(self, server):
    server_state = server.model_dict[self.model_name].state_dict()
    self.model.load_state_dict(server_state, strict=False)

    
  def compute_weight_update(self, epochs=1, loader=None, lambda_fedprox=0.0):
    train_stats = train_op(self.model, self.loader if not loader else loader, self.optimizer, epochs, lambda_fedprox=lambda_fedprox)
    return train_stats


  def predict_logit(self, x):
    """Softmax prediction on input"""
    self.model.train()

    with torch.no_grad():
      y_ = self.model(x)

    return y_


  def extract_features_and_compute_scores(self, loader_loc, loader_pub, loader_distill, lambda_reg=0.01, eps_delt=None):

    self.model.eval()

    ### Train Scoring Model
    local_features = torch.cat([self.model.extract_features(x.cuda()).detach()  for x, _ in loader_loc])
    public_features = torch.cat([self.model.extract_features(x_pub.cuda()).detach() for x_pub, _ in loader_pub])

    X = torch.cat([local_features, public_features]).cpu().numpy()
    y = np.concatenate([np.zeros(local_features.shape[0]), np.ones(public_features.shape[0])])

    norm = np.linalg.norm(X, axis=1).max()
    X_normalized = X / norm

    clf = LogisticRegression(penalty="l2", C=1/lambda_reg, max_iter=1000).fit(X_normalized, y)

    ### Add Differential Privacy
    n = X.shape[0]
    sensitivity = 2/(n*lambda_reg)

    if eps_delt is not None:
      epsilon, delta = eps_delt

      sig2 = 2*np.log(1.25/delta)*sensitivity**2/epsilon**2

      clf.coef_ = clf.coef_ + np.sqrt(sig2) * np.random.normal(size=clf.coef_.shape)
      clf.intercept_ = clf.intercept_ + np.sqrt(sig2) * np.random.normal(size=clf.intercept_.shape)

    ### Compute Scores
    scores = []
    idcs = []
    for (x, _), idx in loader_distill:
      x = self.model.extract_features(x.cuda()).detach()/norm
      scores += [torch.Tensor(clf.predict_proba(x.cpu().numpy())[:,0])]
      idcs += [idx]

    argidx = torch.argsort(torch.cat(idcs, dim=0))
    scores =  torch.cat(scores, dim=0)[argidx].detach()+1e-8


    return scores

    
 

class Server(Device):
  def __init__(self, model_names, optimizer_fn, loader, unlabeled_loader, num_classes=10):
    super().__init__(loader)
    self.distill_loader = unlabeled_loader

    self.model_dict = {model_name : partial(model_utils.get_model(model_name)[0], num_classes=num_classes)().to(device) for model_name in model_names}

    self.parameter_dict = {model_name : {key : value for key, value in model.named_parameters()} for model_name, model in self.model_dict.items()}

    self.optimizer_fn = optimizer_fn
    self.optimizer_dict = {model_name : self.optimizer_fn(model.parameters()) for model_name, model in self.model_dict.items()}  

    self.models = list(self.model_dict.values())


  def evaluate_ensemble(self, loader=None):
    return eval_op_ensemble(self.models, self.loader if not loader else loader)


  def evaluate_individual(self, loader=None):
    return {name : eval_op_ensemble([model], self.loader if not loader else loader) for name, model in self.model_dict.items()}
     

  def select_clients(self, clients, frac=1.0):
    return random.sample(clients, int(len(clients)*frac)) 
    

  def aggregate_weight_updates(self, clients):
    unique_client_model_names = np.unique([client.model_name for client in clients])

    for model_name in unique_client_model_names:
      reduce_average(target=self.parameter_dict[model_name], sources=[client.W for client in clients if client.model_name == model_name])




  def distill(self, clients, epochs=1, mode="mean_probs", num_classes=10):
    
    for model_name in self.model_dict:

      print("Distilling {} ...".format(model_name))

      model = self.model_dict[model_name]
      optimizer = self.optimizer_dict[model_name]

      model.train()  

      acc = 0
      for ep in range(epochs):
        running_loss, samples = 0.0, 0
        for (x, _), idx in tqdm(self.distill_loader):   
          x = x.to(device)


          if mode == "mean_logits":
            y = torch.zeros([x.shape[0], num_classes], device="cuda")
            for i, client in enumerate(clients):
              y_p = client.predict_logit(x)
              y += (y_p/len(clients)).detach()

            y = nn.Softmax(1)(y)


          if mode == "weighted_logits_precomputed":
            y_w = torch.zeros([x.shape[0], num_classes], device="cuda")
            w = torch.zeros([x.shape[0], 1], device="cuda")
            for i, client in enumerate(clients):
              y_p = client.predict_logit(x)
              weight = client.scores[idx].reshape(-1,1).cuda()

              y_w += (y_p*weight).detach()
              w += weight.detach()

            y = nn.Softmax(1)(y_w / w)          

   
          optimizer.zero_grad()

          y_ = nn.LogSoftmax(1)(model(x))

          loss = torch.nn.KLDivLoss(reduction="batchmean")(y_, y.detach())


          running_loss += loss.item()*y.shape[0]
          samples += y.shape[0]

          loss.backward()
          optimizer.step()  


    return {"loss" : running_loss / samples, "epochs" : ep}







def train_op(model, loader, optimizer, epochs, lambda_fedprox=0.0):
    model.train() 

    W0 = {k : v.detach().clone() for k, v in model.named_parameters()}

    running_loss, samples = 0.0, 0
    for ep in range(epochs):
      for x, y in loader:   
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()

        loss = nn.CrossEntropyLoss()(model(x), y)

        if lambda_fedprox > 0.0:
          loss += lambda_fedprox * torch.sum((flatten(W0).cuda()-flatten(dict(model.named_parameters())).cuda())**2)

        running_loss += loss.item()*y.shape[0]
        samples += y.shape[0]

        loss.backward()
        optimizer.step()  

    return {"loss" : running_loss / samples}


def eval_op(model, loader):
    model.train()
    samples, correct = 0, 0

    with torch.no_grad():
      for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        y_ = model(x)
        _, predicted = torch.max(y_.detach(), 1)
        
        samples += y.shape[0]
        correct += (predicted == y).sum().item()

    return {"accuracy" : correct/samples}


def eval_op_ensemble(models, loader):
    for model in models: 
        model.train()


    samples, correct = 0, 0

    with torch.no_grad():
      for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        y_ = torch.mean(torch.stack([model(x) for model in models], dim=0), dim=0)
        _, predicted = torch.max(y_.detach(), 1)
        
        samples += y.shape[0]
        correct += (predicted == y).sum().item()

    return {"accuracy" : correct/samples}


def reduce_average(target, sources):
  for name in target:
      target[name].data = torch.mean(torch.stack([source[name].detach() for source in sources]), dim=0).clone()

def flatten(source):
  return torch.cat([value.flatten() for value in source.values()])

def copy(target, source):
  for name in target:
    target[name].data = source[name].detach().clone()
