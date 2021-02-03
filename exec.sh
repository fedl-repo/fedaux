

cmdargs=$1


hyperparameters='[{
    "random_seed" : [2],

    "dataset" : ["cifar10"], 
    "distill_dataset" : ["cifar100_distill"],
    "models" : [{"resnet8" : 20}],


    "participation_rate" : [1.0],

    "alpha" : [0.01],


    "communication_rounds" : [10],
    "local_epochs" : [1],
    "distill_epochs" : [1],
    "n_distill_frac" : [0.8], 


    "batch_size" : [32],
    "distill_batch_size" : [128],
    "local_optimizer" : [ ["Adam", {"lr": 0.001}]],
    "distill_optimizer" : [["Adam", {"lr": 0.00005}] ],


    "aggregation_mode" : ["FedAUX"],
    "save_scores" : [false],


    "pretrained" : [null],

    "eps_delt" : [[0.1, 1e-5]],
    "lambda_reg_score" : [0.01],

    "lambda_fedprox" : [0.001], 



    "save_model" : [null],
    "log_frequency" : [-100],
    "log_path" : ["trash/"]}]

'



RESULTS_PATH="results/"
DATA_PATH="data/"
CHECKPOINT_PATH="checkpoints/"

python -u code/federated_learning.py --hp="$hyperparameters" --RESULTS_PATH="$RESULTS_PATH" --DATA_PATH="$DATA_PATH" --CHECKPOINT_PATH="$CHECKPOINT_PATH" $cmdargs











