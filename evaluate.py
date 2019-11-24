import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from util import create_bagged_predictor
from copy import deepcopy
"""
def class_prob_avg_fn(class_probabilities):
    K = len(class_probabilities[0])
    return [np.mean([x[i] for x in class_probabilities]) for i in range(K)]

def class_majority_vote(class_probabilities):
    K = len(class_probabilities[0])
    votes = [np.argmax(pr) for pr in class_probabilities]
    num_votes_per_class = [len([i for i in votes if i == v]) for v in range(K)]
    return [1 if i == np.argmax(num_votes_per_class) else 0 for i in range(K)]
"""

def calculate_oob(data_x, data_y, model, model_param, num_feats_per_bootstrap='all', seed=35901):
    if(num_feats_per_bootstrap=='all'):
        num_feats_per_bootstrap = data_x.shape[1]
    predictor_fn, models, not_seen, seen = create_bagged_predictor(data_x, data_y, model, model_param, num_feats_per_bootstrap, seed)
    B = len(models)
   
    train_errs = []
    test_errs = []
    for i in range(B):
        test_data = not_seen[i]
        m = models[i]
        m.fit(data_x.loc[seen[i]], data_y.loc[seen[i]])
        train_err = metrics.accuracy_score(m.predict(data_x.loc[seen[i]]), data_y.loc[seen[i]])
        train_errs.append(train_err)
        test_err = metrics.accuracy_score(models[i].predict(data_x.loc[test_data]), data_y.loc[test_data])
        test_errs.append(test_err)
    
    unseen_points_dict = dict()
    for point in data_x.index:
        lst = [i for i in range(B) if point in not_seen[i]]
        if(len(lst) > 0):
            unseen_points_dict[point] = deepcopy(lst)

    n_correct = 0.0
    n_total = 0
    for pt, model_inds in unseen_points_dict.items():
        predicted_val = predictor_fn(data_x.loc[pt].values.reshape(1, -1),  model_inds)
        n_total += 1
        if(predicted_val == data_y.loc[pt]):
            n_correct += 1

    test_oob = n_correct/n_total


    seen_points_dict = dict()
    for point in data_x.index:
        lst = [i for i in range(B) if point in np.unique(seen[i])]
        if(len(lst) > 0):
            seen_points_dict[point] = deepcopy(lst)

    n_correct = 0.0
    n_total = 0
    for pt, model_inds in seen_points_dict.items():
        predicted_val = predictor_fn(data_x.loc[pt].values.reshape(1, -1), model_inds)
        n_total += 1
        if(predicted_val == data_y.loc[pt]):
            n_correct += 1

    train_oob = n_correct/n_total
    return {'test oob':test_oob, 'train oob': train_oob, 'train errors': train_errs, 'test errors': test_errs, 'predictor': predictor_fn}



