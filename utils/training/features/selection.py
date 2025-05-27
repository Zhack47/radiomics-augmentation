import copy
import pandas
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
    m = CoxnetSurvivalAnalysis()
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