# Federated Learning with Auxiliary Daata


## Usage

1.) In `exec.sh` define paths

	RESULTS_PATH="results/"
	DATA_PATH="/path/to/where/you/store/your/datasets"
	CHECKPOINT_PATH="checkpoints/"
  
2.) and set the hyperparameters
  
    hyperparameters="[{...}]"

3.) Run via

    bash exec.sh

      
 ## Hyperparameters
 
 ### Task
- `"dataset"` : Choose from `["cifar10", "cifar100"]`
- `"distill_dataset"` : Choose from `["stl10", "cifar100_distill"]`,
- `"models"` : Choose from `[{"resnet8" : n}, {mobilenetv2" : n}, {"shufflenet" : n}, {"resnet8" : n1, mobilenetv2" : n2, "shufflenet" : n3}]` where n is the number of clients

### Federated Learning Environment


- `"participation_rate"` : Fraction of Clients which participate in every Communication Round
- `"alpha"` : Dirichlet Data Heterogeneity Parameter
- `"communication_rounds"` : Total number of communication rounds
- `"local_epochs"` : Local training epochs at every client
- `"distill_epochs"` : Number of epochs used for distillation
- `"n_distill_frac"` : Fraction of the auxiliary data that is used for distillation 

### Optimization Parameters

- `"batch_size"` : Batch-size used by the Clients
- `"distill_batch_size"` : Batch-size used for distillation at the server
- `"local_optimizer"` : The Optimizer used for local training
- `"distill_optimizer"` : The optimizer used for Distillation

- `"aggregation_mode"` : Choose from "FedAVG" (Federated Averaging), "FedDF" (Federated Ensemble Distillation), "FedAUX" (Federated Learning with Auxiliary Data), "+P" uses pretrained models, "+L" uses linear evaluation
- `"save_scores"` : Log the certainty scores computed by FedAUX
- `"pretrained"` : Load a pretrained model from the /checkpoints directory according to the auxiliary data that is used, by default will load "<"model_name">_<"distill_dataset">.pth" (the provided checkpoints were obtained by self-supervised pre-training, code available at: https://github.com/leftthomas/SimCLR)

- `"eps_delt"` : (epsilon,delta)-Privacy Parameter of FedAUX
- `"lambda_reg_score"` : Regularization Term of the ERM
- `"lambda_fedprox"` : Regularization of the Proximity Term in FedPROX


### Logging 
- `"log_frequency"` : Number of communication rounds after which results are logged and saved to disk
- `"log_path"` : e.g. "results/experiment1/"


Run multiple experiments by listing different configurations, e.g.

	`"alpha" : [0.01, 0.1, 1.0]`.

## Logging and Visualizing 
In `federated_learning.py`, calling

	xp.log(dict)

will save experiment results under given keys. Every experiment produces a summary which is stored in the directory specified in `"log_path"`. You can import all experiments stored in a certain directory via

	import experiment_manager as xpm
	list_of_experiments = xpm.get_list_of_experiments(path)
	
`list_of_experiments` contains `Experiment` objects with the hyperparameters and the results

	xp.hyperparameters
	xp.results
	
of the respective experiments.
	

