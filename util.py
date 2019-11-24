import pandas as pd
import random as rand
import numpy as np
from copy import deepcopy

def create_bagged_predictor(train_x, train_y, model, model_params, p, seed=35901):
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
    train_data = pd.concat([train_x, train_y], axis=1)
    response_var = train_y.name
    features = [x for x in train_data.columns if x != response_var]


    # Create Bootstrap Samples

    randomly_sampled_data = rand.choices(train_data.index, k=n*B)
    boot_data_sets = [randomly_sampled_data[(i-1)*n:i*n] for i in range(1,B+1)] #setting this to k=2*n improves performance
    boot_feat_sets = [rand.sample(features, p) for __ in range(B)]

    # Train models on bootstrapped data/feature sets
    Models = []
    not_seen = []
    seen_data = []
    for i in range(B):
        model_obj = deepcopy(model)
        model_obj.set_params(**model_params)
        data = train_data.loc[boot_data_sets[i]]
        seen_data.append(boot_data_sets[i])
        unique_vals = np.unique(boot_data_sets[i])
        not_seen.append([x for x in train_data.index if x not in unique_vals])
        feats = boot_feat_sets[i]
        model_obj.fit(train_x.loc[boot_data_sets[i]], train_y.loc[boot_data_sets[i]] )
        Models.append(model_obj) 

    # Create and return predictor function which bags these models
    def predictor_fn(X, models_to_consider='all'):
        if(models_to_consider == 'all'):
            models_to_consider = range(B)

        votes = [Models[i].predict(X) for i in models_to_consider]
        return mode(votes)


    return predictor_fn, Models, not_seen, seen_data



def mode(lst_of_votes):
    #tie breaking: smaller class index
    frequencies = dict()
    for v in lst_of_votes:
        v = v[0]
        if(v not in frequencies.keys()):
            frequencies[v] = 0
        else:
            frequencies[v] += 1
    return max(frequencies, key=lambda key: frequencies[key])

