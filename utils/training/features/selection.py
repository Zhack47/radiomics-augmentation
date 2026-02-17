import copy
import heapq
import pandas
import random
import pickle
import numpy as np
from tqdm import tqdm
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis


def ci_classif(X,Y):
    m = CoxnetSurvivalAnalysis()
    print(X.shape)
    print(Y)
    #del X["RFS"]
    #del X["Relapse"]
    #del X["Patient ID"]
    y = Surv.from_arrays(Y[:, 0], Y[:, 1])
    m.fit(X, y)
    out = m.predict(X)
    score, _, _, _, _ = concordance_index_censored(Y[:, 0]==1, Y[:, 1], out)
    print(score)
    return score


def fit_and_score_features(X, y):
    scores = {}
    y = Surv.from_arrays(y["Relapse"], y["RFS"])
    #m = CoxnetSurvivalAnalysis()
    m = CoxPHSurvivalAnalysis()
    for col in tqdm(X.columns):
        Xj = X[col].values.reshape(-1,1)
        m.fit(Xj, y)
        scores[col] = m.score(Xj, y)
    df_scores = pandas.Series(scores, index=X.columns).sort_values(ascending=False)
    return df_scores

def make_scores(X):
    x_classify_copy = copy.deepcopy(X)
    y_classify_copy = copy.deepcopy(X)
    del x_classify_copy["Patient ID"]
    del y_classify_copy["Patient ID"]
    x_classify_copy.fillna(0, inplace=True)
    for i in x_classify_copy.columns:
        if i in ["RFS", "Relapse"]:
            del x_classify_copy[i]
        else:
            del y_classify_copy[i]
    #scores = pickle.load("../../data/coeffs.pkl")
    scores = fit_and_score_features(x_classify_copy, y_classify_copy)
    #pickle.dump(scores, open("../../data/coeffs.pkl", "wb"))
    return scores


def get_duplicates(dataframe):
    columns_to_remove = []
    list_columns = copy.deepcopy(dataframe.columns)
    i = 1
    for column1 in list_columns:
        for column2 in list_columns[i:]:
            if dataframe[column1].equals(dataframe[column2]):
                columns_to_remove.append(column1)
                break
        i+=1
    return columns_to_remove


def select_k_transforms(names, nb_image_transforms, nb_mask_transforms):
    available_image_transforms = set()
    available_mask_transforms = set()
    for name in names:
        _, im_tr, mask_tr = name.split("_")
        available_image_transforms.add(im_tr)
        available_mask_transforms.add(mask_tr)
    select_im_tr = random.sample(list(available_image_transforms), nb_image_transforms)
    select_mask_tr = random.sample(list(available_mask_transforms), nb_mask_transforms)
    return select_im_tr, select_mask_tr

def get_noncorrelated_features(dataframe, target, threshold):
    m = CoxnetSurvivalAnalysis()
    #m = CoxPHSurvivalAnalysis()
    non_correlated = []
    for col in tqdm(dataframe.columns):
        Xc = dataframe[col].values.reshape(-1,1)
        m.fit(Xc, target)
        score = m.score(Xc, target)
        if score < threshold:
            non_correlated.append(col)
    return non_correlated

def get_best_features(dataframe, target, nb_features):
    features_scores = {}
    m = CoxnetSurvivalAnalysis()
    #m = CoxPHSurvivalAnalysis()
    for col in tqdm(dataframe.columns):
        Xc = dataframe[col].values.reshape(-1,1)
        m.fit(Xc, target)
        score = m.score(Xc, target)
        features_scores[col] = score
    best_features = heapq.nlargest(nb_features, features_scores, key=features_scores.get)
    return best_features