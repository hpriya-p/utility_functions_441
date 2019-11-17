import pandas as pd
import numpy as np
from util import create_bagged_predictor, create_train_model_fn

def evaluate_bagged_model(train_csv_path, test_csv_path, response_var, model, model_param, p, verbose=False, seed=35901):
    train_model_fn = create_train_model_fn(model, model_param)
    predictor_fn = create_bagged_predictor(train_csv_path, response_var, train_model_fn, p, verbose, seed)

    test_data = pd.read_csv(test_csv_path, header=0)
    features = [x for x in test_data.columns if x != response_var]
    classes = np.unique(pd.read_csv(train_csv_path, header=0)[response_var].values)
    n_t = test_data.shape[0]
    predictions = [np.argmax(predictor_fn(test_data.iloc[i])) for i in range(n_t)]
    num_correct = sum([1 if classes[predictions[i]] == test_data[response_var].iloc[i] else 0 for i in range(n_t)])
    return num_correct/n_t
    
    
