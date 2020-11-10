from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_validate
from genetic import GA
import numpy as np
from tqdm import tqdm

def kfold_verify(X, y, model):
    kfcv = KFold(shuffle=True)
    cv_scores = cross_validate(model, X, y, cv=kfcv, scoring='neg_root_mean_squared_error' )
    return -(np.mean(cv_scores['test_score']))         # Lower is better


def eval_weights(weight_arr, X, y):
    """
    Parameters:-
    weight_arr: 1D array, indicating the weights for each attributes
    Returns:-
    The cross-validation error rate for given weights
    """
    model = KNeighborsRegressor(10, metric='wminkowski', metric_params={'w':weight_arr})
    return kfold_verify(X, y, model)

def knn_optimize(X, y):
    pop_size = (100, X.shape[1])
    genetic_obj = GA(pop_size, 10, 50)
    genetic_obj.initialize(X,y)

    epochs = 10
    print('Optimising Attribute weights')
    for i in tqdm(range(epochs)):
        fit_score = genetic_obj.calc_fitness(eval_weights)
        next_gen_parents = genetic_obj.get_parents(fit_score)
        next_gen = genetic_obj.crossover()
        genetic_obj.next_gen()
    final_attr_weights = genetic_obj.get_optimal_genes(eval_weights)
    return final_attr_weights
