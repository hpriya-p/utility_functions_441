import pandas as pd
import random as rand
import numpy as np

def create_bagged_predictor(train_x, train_y, train_model_fn, p, verbose=False, seed=35901):
    """
    Returns predictor_fn(X), a bagged predictor function. Requires the following inputs:
        * train_x, train_y: pandas DataFrames
        * response_var: the name of the response variable (as it appears in the training data csv file)
        * train_model_fn: a function for training the model. Should be created using create_train_model_fn
        * p : number of features bootstrapped samples should choose
    """
    #first row of csv should be feature names
    rand.seed(seed)
   
    # CONSTANTS
    n = train_x.shape[0]        # sample size 
    B = 50                      # number of bootstrap samples
    p = p                       # number of features per bootstrap sample

    # Preliminaries
    train_data = train_x
    response_var = train_y.columns[0]
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
        Models.append(train_model_fn(data[feats], data[response_var])) 

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



def create_train_model_fn(model_obj, params_dict):
    """
    Given a Scikit-learn model object and a dictionary of parameters + values,
    returns a train_model_fn which can be provided as input to create_bagged_predictor
    """

    def train_model_fn(X, y):
        model_obj.set_params(params_dict)
        model_obj.fit(X, y)
        return model_obj
    return train_model_fn

