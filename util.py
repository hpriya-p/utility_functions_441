import pandas as pd
import random as rand
import numpy as np

def create_bagged_predictor(train_csv_path, response_var, train_model_fn, p, verbose=False, seed=35901):
    #first row of csv should be feature names
    rand.seed(seed)
   
    # CONSTANTS
    n = train_data.shape[0]     # sample size 
    B = 50                      # number of bootstrap samples
    p = p                       # number of features per bootstrap sample

    # Preliminaries
    train_data = pd.read_csv(train_csv_path, header=0)
    features = [x for x in train_data if x != response_var]
    if(verbose): print("features: " + ", ".join(features))

    # Create Bootstrap Samples
    boot_data_sets = [rand.sample(train_data.index, n, with_replacement=True) for __ in range(B)]
    boot_feat_sets = [rand.sample(features, p, with_replacement=True) for __ in range(B)]

    # Train models on bootstrapped data/feature sets
    Models = []
    for i in range(B):
        data = boot_data_sets[i]
        feats = boot_featsets[i]
        Models.append(train_model_fn(data[feats], data[response_var])) #HP note: not sure if training will return a model obj.

    # Create and return predictor function which bags these models
    def predictor_fn(X):
        probabilities = []
        for i in range(B):
            try:
                probabilities.append(Models[i].predict_proba(X))
            except:
                print("ERROR: could not predict model")
                return -1

        return np.mean(probabilities)

    return predictor_fn

def create_train_model_fn
