## Inspired from https://github.com/Lrebaud/ICARE/blob/main/notebook/reproducing_HECKTOR2022.ipynb

import numpy as np
import copy
import warnings
import pandas as pd
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sklearn.model_selection import StratifiedKFold
from sksurv.svm import FastSurvivalSVM
from sksurv.linear_model import CoxnetSurvivalAnalysis
from icare.survival import BaggedIcareSurvival
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from scipy.spatial import distance

from tqdm import tqdm


warnings.filterwarnings("ignore")
augmented=True
model_name = "FS_SVM" # "icare"
if augmented:
    csv_file = open(f"../csvs/Perf_Hecktor_augmented_{model_name}.csv", "w")
else:
    csv_file = open(f"../csvs/Perf_Hecktor_base_{model_name}.csv", "w")
csv_file.write("nb_variables,CI_train,CI_test,cdAUC_train,cdAUC_test\n")
if augmented:
    augmented_radiomics = pd.read_csv("../csvs/Hecktor22_AugmentedRadiomics.csv",
                                      dtype={"Elongation": float, "Flatness": float,
                                           "MajorAxisLength": float, "MinorAxisLength": float,
                                           "LeastAxisLength": float})
    for column in augmented_radiomics.columns:
        if column.split("_")[-1] in ["Elongation", "Flatness", "LeastAxisLength", "MinorAxisLength", "MajorAxisLength"]:
            augmented_radiomics[column] = augmented_radiomics[column].apply(lambda x: np.real(np.complex64(x)))

    augmented_radiomics["Patient Name"] = augmented_radiomics["Patient ID"].apply(lambda x: x.split("_")[0])
else:
    normal_radiomics = pd.read_csv("../csvs/Hecktor22_Radiomics.csv", index_col="Patient ID")
    normal_radiomics["Patient Name"] = normal_radiomics.index


if augmented:
    endpoints = pd.read_csv("../csvs/hecktor2022_endpoint_training.csv")  # Maybe try using index again
    clinical = pd.read_csv("../csvs/hecktor2022_clinical_info_training.csv")  # Maybe try using index again
else:
    endpoints = pd.read_csv("../csvs/hecktor2022_endpoint_training.csv", index_col="PatientID")
    clinical = pd.read_csv("../csvs/hecktor2022_clinical_info_training.csv", index_col="PatientID")

clinical["Weight"] = [np.nanmedian(clinical["Weight"]) if np.isnan(x) else x for x in clinical["Weight"]]
clinical["Gender"] = [-1 if x == "M" else 1 for x in clinical["Gender"]]
clinical["Tobacco"] = [1 if x == 1. else -1 if x ==0. else 0 for x in clinical["Tobacco"]]
clinical["Alcohol"] = [1 if x == 1. else -1 if x ==0. else 0 for x in clinical["Alcohol"]]
clinical["Performance status"] = [-1 if np.isnan(x) else x for x in clinical["Performance status"]]
clinical["Surgery"] = [1 if x == 1. else -1 if x ==0. else 0 for x in clinical["Surgery"]]
clinical["HPV status (0=-, 1=+)"] = [1 if x == 1. else -1 if x ==0. else 0 for x in clinical["HPV status (0=-, 1=+)"]]
if augmented:
    df_train = augmented_radiomics
else:
    df_train = normal_radiomics
# df_train = clinical



if augmented:
    def get_ridiculous_augments(df: pd.DataFrame, thresh):
        remove = []
        for pname in df["Patient Name"].unique():
            true = copy.deepcopy(df[df["Patient ID"]==f"{pname}_Identity_Identity"])
            augs = copy.deepcopy(df[(df["Patient Name"]==pname) & (df["Patient ID"]!=f"{pname}_Identity_Identity")])
            print("_____________")
            print(true["Patient ID"])
            print(augs["Patient ID"])
            print("_____________")
            del true["Patient ID"]
            del true["Patient Name"]
            for aug_id in augs["Patient ID"]:
                aug = copy.deepcopy(augs[augs["Patient ID"]==aug_id])
                del aug["Patient Name"]
                del aug["Patient ID"]
                print(aug.values.shape)
                print(true.values.shape)
                aug.fillna(0)
                true.fillna(0)
                aug_values = aug.values[0]
                true_values = true.values[0]
                if distance.euclidean(aug_values, true_values, abs(1 / (true_values) + .1))) / len(aug_values) < thresh:
                    remove.append(aug_id)
        return remove
    to_remove = get_ridiculous_augments(df_train,10)
    df_train = df_train[~df_train["Patient ID"].isin(to_remove)]        

    print(df_train.shape)
    df_train = pd.merge(df_train, clinical, left_on="Patient Name", right_on="PatientID")
    print(df_train.shape)
    df_train = pd.merge(df_train, endpoints, left_on="Patient Name", right_on="PatientID")
    print(df_train.shape)
    

    df_train = df_train.set_index("Patient ID")
    #del df_train["Patient Name"]
    #del df_train["Patient ID"]
    del df_train["PatientID_y"]
    del df_train["PatientID_x"]
else:
    df_train = pd.merge(df_train, clinical, left_index=True, right_index=True)
    df_train = pd.merge(df_train, endpoints, left_index=True, right_index=True)

y_train = Surv.from_arrays(event=df_train['Relapse'].values,
                           time=df_train['RFS'].values)
kfold = StratifiedKFold(5, random_state=np.random.randint(0, 1e9), shuffle=True)

df_train = df_train.fillna(0)


ids = np.unique(df_train["Patient Name"].values)


# Removing duplicate columns
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

duplicate_columns = get_duplicates(df_train)
for c in duplicate_columns:
    if c not in ["RFS", "Relapse", "Patient ID"]:
        del df_train[c]
print(df_train.shape)

if augmented:
    censored = [df_train[df_train.index==f"{x}_Identity_Identity"]["Relapse"] for x in ids]
else:
    censored = df_train["Relapse"]
for thr in tqdm(range(1,df_train.values.shape[1],1)):
    ci_avg_test = 0.
    ci_avg_train = 0.
    cdauc_avg_test=0.
    cdauc_avg_train=0.
    for tr_ids, ts_ids in kfold.split(ids, censored):
        print("###############################################")
        train_ids = ids[tr_ids]
        test_ids = ids[ts_ids]
        scaler = StandardScaler()
        X_train_local = df_train[df_train["Patient Name"].isin(train_ids)]
        X_test_local = df_train[df_train["Patient Name"].isin(test_ids)]

        Y_train_local = Surv.from_arrays(df_train[df_train["Patient Name"].isin(train_ids)]["Relapse"],
                                            df_train[df_train["Patient Name"].isin(train_ids)]["RFS"])
        Y_test_local = Surv.from_arrays(df_train[df_train["Patient Name"].isin(test_ids)]["Relapse"],
                                            df_train[df_train["Patient Name"].isin(test_ids)]["RFS"])


        banned_features = ["RFS", "Relapse", "Task 1", "Task 2", "CenterID", "Patient Name"]
        for banned_feature in banned_features:
            try:
                del X_train_local[banned_feature]
                del X_test_local[banned_feature]
            except KeyError:
                pass
        
        def f_uci(X,Y):
            Y = Surv.from_arrays(event=[x[0] for x in Y], time=[x[1] for x in Y])
            scores = []
            pvals = []
            for col_nb in tqdm(range(X.shape[1])):
                fs_model = CoxnetSurvivalAnalysis()
                fs_model.fit(X[:,col_nb].reshape(-1, 1), Y)
                corr_score = concordance_index_censored(Y["event"], Y["time"],
                                                            fs_model.predict(X[:, col_nb].reshape(-1, 1)))
                scores.append(corr_score[0])
                pvals.append(1/corr_score[0])
                scores_cache, pvals_cache = copy.deepcopy(scores), copy.deepcopy(pvals)
            return scores, pvals
        
        try:
            selector
        except NameError:
            selector = SelectKBest(score_func=f_uci, k=thr)
            X_train_selected = selector.fit(X_train_local.values, Y_train_local)
        
        # The _get_support_mask function uses k and is called by transform to select variables
        # We set k and perform selection on already computed UCI scores
        print(selector.k, thr)
        selector.set_params(k=thr)
        print(selector.k, thr)
        X_train_selected = selector.transform(X_train_local)
        X_test_selected = selector.transform(X_test_local)

        selected_features = X_train_local.columns[selector.get_support()]
        f_scores = selector.scores_[selector.get_support()]
        #print(f"Selected Features: {selected_features}")
        #print(f"F-Scores: {f_scores}")
        X_train_local = X_train_local[selected_features]
        X_test_local = X_test_local[selected_features]
        
        
        X_train_local_np = scaler.fit_transform(X_train_local)
        X_test_local_np = scaler.transform(X_test_local)

        if model_name == "icare":
            model = BaggedIcareSurvival(n_estimators=100,
                                    parameters_sets=None,
                                    aggregation_method='median',
                                    n_jobs=-1)
        elif model_name == "FS_SVM":
            model = FastSurvivalSVM()
        model.fit(X_train_local, Y_train_local)
        train_pred = model.predict(X_train_local)
        test_pred = model.predict(X_test_local)

        # C-Index
        ci_train = concordance_index_censored(Y_train_local["event"], Y_train_local["time"], train_pred)
        ci_test = concordance_index_censored(Y_test_local["event"], Y_test_local["time"], test_pred)
        ci_avg_train+=ci_train[0]
        ci_avg_test+=ci_test[0]
        print(ci_train[0], ci_test[0])
        
        # cdAUC
        time_points_train = np.arange(Y_train_local["time"][np.argpartition(Y_train_local["time"],5)[5]],
                                                        Y_train_local["time"][np.argpartition(Y_train_local["time"],-5)[-5]], 50)
        time_points_test = np.arange(Y_test_local["time"][np.argpartition(Y_test_local["time"],5)[5]],
                                                        Y_test_local["time"][np.argpartition(Y_test_local["time"],-5)[-5]], 50)
        cdauc_train = cumulative_dynamic_auc(Y_train_local, Y_train_local, train_pred, times=time_points_train)[1]
        cdauc_test = cumulative_dynamic_auc(Y_train_local, Y_test_local, test_pred, times=time_points_test)[1]
        cdauc_avg_train+=cdauc_train
        cdauc_avg_test+=cdauc_test
        print(cdauc_train, cdauc_test)

    print(f"Average train CI:{ci_avg_train/5}")
    print(f"Average test CI:{ci_avg_test/5}")
    print(f"Average train cdAUC:{cdauc_avg_train/5}")
    print(f"Average test cdAUC:{cdauc_avg_test/5}")
    csv_file.write(f"{thr},{ci_avg_train/5},{ci_avg_test/5},{cdauc_avg_train/5},{cdauc_avg_test/5}\n")