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
    print(n)
    B = 50                      # number of bootstrap samples
    p = p                       # number of features per bootstrap sample

    # Preliminaries
    train_data = pd.concat([train_x, train_y], axis=1)
    response_var = train_y.name
    features = [x for x in train_data.columns if x != response_var]
    if(verbose): print("features: " + ", ".join(features))

    # Create Bootstrap Samples

    randomly_sampled_data = rand.choices(train_data.index, k=n*B)
    boot_data_sets = [randomly_sampled_data[(i-1)*n:i*n] for i in range(1,B)] #setting this to k=2*n improves performance

    boot_feat_sets = [rand.sample(features, p) for __ in range(B)]

    # Train models on bootstrapped data/feature sets
    Models = []
    for i in range(B):
        data = train_data.loc[boot_data_sets[i]]
        feats = boot_feat_sets[i]
        Models.append(train_model_fn(data[feats], data[response_var])) 

    # Create and return predictor function which bags these models
    def predictor_fn(X, prob_aggr_fn = np.mean):
        probabilities = []
        for i in range(B):
            probabilities.append(list(Models[i].predict_proba(X[boot_feat_sets[i]].values.reshape(1, -1))[0]))

        return prob_aggr_fn(probabilities)

    return predictor_fn, Models



def create_train_model_fn(model_obj, params_dict):
    """
    Given a Scikit-learn model object and a dictionary of parameters + values,
    returns a train_model_fn which can be provided as input to create_bagged_predictor
    """

    def train_model_fn(X, y):
        model_obj.set_params(**params_dict)
        model_obj.fit(X, y)
        return model_obj
    return train_model_fn

