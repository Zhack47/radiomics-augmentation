import copy
import numpy as np
from tqdm import tqdm
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis


def f_uci(X,Y):
            Y = Surv.from_arrays(event=[x[0] for x in Y], time=[x[1] for x in Y])
            scores = []
            pvals = []
            for col_nb in tqdm(range(X.shape[1]), ncols=64):
                fs_model = CoxnetSurvivalAnalysis()
                fs_model.fit(X[:,col_nb].reshape(-1, 1), Y)
                corr_score = concordance_index_censored(Y["event"], Y["time"],
                                                            fs_model.predict(X[:, col_nb].reshape(-1, 1)))
                scores.append(corr_score[0])
                pvals.append(1/corr_score[0])
            return scores, pvals


def get_duplicates(dataframe):
    columns_to_remove = []
    list_columns = [i for i in copy.deepcopy(dataframe.columns) if i not in ["RFS", "Relapse", "Patient ID", "Patient Name"]]
    i = 1 
    for column1 in list_columns:
        for column2 in list_columns[i:]:
            if dataframe[column1].equals(dataframe[column2]):
                columns_to_remove.append(column1)
                break
        i+=1
    return columns_to_remove