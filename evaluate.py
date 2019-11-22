import pandas as pd
import numpy as np
from util import create_bagged_predictor, create_train_model_fn

def class_prob_avg_fn(class_probabilities):
    K = len(class_probabilities[0])
    return [np.mean([x[i] for x in class_probabilities]) for i in range(K)]

def class_majority_vote(class_probabilities):
    print(class_probabilities)
    K = len(class_probabilities[0])
    votes = [np.argmax(pr) for pr in class_probabilities]
    num_votes_per_class = [len([i for i in votes if i == v]) for v in range(K)]
    return [1 if i == np.argmax(num_votes_per_class) else 0 for i in range(K)]

def evaluate_bagged_model(train_x, train_y, test_x, test_y, model, model_param, p, prob_aggr_fn=class_prob_avg_fn, verbose=False, seed=35901):
    train_model_fn = create_train_model_fn(model, model_param)
    predictor_fn = create_bagged_predictor(train_x, train_y, train_model_fn, p, verbose, seed)

    test_data = pd.concat([test_x, test_y], axis=1)
    response_var = test_y.name
    features = [x for x in test_data.columns if x != response_var]
    classes = np.unique(train_y.values)
    n_t = test_data.shape[0]
    predicted_probs = [predictor_fn(test_data.iloc[i], prob_aggr_fn) for i in range(n_t)]
    predictions = [classes[np.argmax(predicted_probs[i])] for i in range(n_t)]
    num_correct = sum([1 if predictions[i] == test_data[response_var].iloc[i] else 0 for i in range(n_t)])
    return num_correct, n_t, predicted_probs, predictions
    
    
