import sys
sys.path.append("../../")
import copy
import pandas
import pickle
import sksurv
import numpy as np
from tqdm import tqdm
from sksurv.svm import FastSurvivalSVM
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sklearn.metrics import make_scorer
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import StratifiedKFold
from utils.training.features.selection import make_scores


# Loading radiomics
hecktor_radiomics = pandas.read_csv("../../csvs/Hecktor22_Radiomics.csv")
# Loading endpoints
endpoints = pandas.read_csv("/media/zhack/Toshiba/drawer/hecktor2022/hecktor2022_endpoint_training.csv")
# Loading clinical data
clinical_data = pandas.read_csv("/media/zhack/Toshiba/drawer/hecktor2022/hecktor2022_clinical_info_training.csv")

# Removing patients not present in the endpoints csv file (Task 1-only patients)
patient_ids_task2 = endpoints["PatientID"]
hecktor_radiomics = hecktor_radiomics[hecktor_radiomics["Patient ID"].isin(patient_ids_task2)]
print(f"Found {len(hecktor_radiomics)} patients with task 2 endpoint")

# Merging the clinical and radiomics data
hecktor_radiomics = pandas.merge(hecktor_radiomics, clinical_data, left_on="Patient ID", right_on="PatientID")
del hecktor_radiomics["PatientID"]
del hecktor_radiomics["Task 1"]
del hecktor_radiomics["Task 2"]
# del hecktor_radiomics["CenterID"]

# Adapting the clinical data taxonomy
hecktor_radiomics["Gender"] = hecktor_radiomics["Gender"].apply(lambda x: 1 if x =="M" else 2)
hecktor_radiomics["Tobacco"] = hecktor_radiomics["Tobacco"].apply(lambda x: 1 if x ==1 else -1 if x==0 else 0)
hecktor_radiomics["Surgery"] = hecktor_radiomics["Surgery"].apply(lambda x: 1 if x ==1 else -1 if x==0 else 0)
hecktor_radiomics["Chemotherapy"] =hecktor_radiomics["Chemotherapy"].apply(lambda x: 1 if x ==1 else -1 if x==0 else 0)
hecktor_radiomics["HPV status (0=-, 1=+)"] = hecktor_radiomics["HPV status (0=-, 1=+)"].apply(lambda x: 1 if x ==1 else -1 if x==0 else 0)
hecktor_radiomics["Performance status"] =hecktor_radiomics["Performance status"].apply(lambda x: x+1 if x in (0,1,2,3,4) else 0)

# Merging the endpoints and data
hecktor_radiomics = pandas.merge(hecktor_radiomics, endpoints,left_on="Patient ID", right_on="PatientID")
del hecktor_radiomics["PatientID"]
patient_ids_new = hecktor_radiomics["Patient ID"]

# Computing the stratified KFold
relapse = hecktor_radiomics["Relapse"]
kfold = StratifiedKFold(n_splits=5, shuffle=True)
splits = kfold.split(patient_ids_new, y=relapse)

cis = []
cis_train = []
# Performing Cross-Validation
for split in splits:
    # Retrieving the training / testing split patient ids
    train_ids = []
    test_ids = []
    train_indices = split[0]
    test_indices = split[-1]
    for tr_idx in train_indices:
        train_ids.append(patient_ids_new[tr_idx])
    for ts_idx in test_indices:
        test_ids.append(patient_ids_new[ts_idx])

    x_train = hecktor_radiomics[hecktor_radiomics["Patient ID"].isin(train_ids)]
    x_test = hecktor_radiomics[hecktor_radiomics["Patient ID"].isin(test_ids)]
    y_train = sksurv.util.Surv.from_dataframe("Relapse", "RFS", x_train)
    y_test = sksurv.util.Surv.from_dataframe("Relapse", "RFS", x_test)

    x_train = x_train.fillna(0)
    x_test = x_test.fillna(0)

    del x_train["RFS"]
    del x_train["Relapse"]
    del x_train["Patient ID"]
    del x_test["RFS"]
    del x_test["Relapse"]
    del x_test["Patient ID"]

    for i in tqdm(range(10)):
        model = RandomSurvivalForest(n_estimators=2000, max_features="sqrt", n_jobs=-1)
        # model = FastSurvivalSVM(max_iter=2)
        
        # Scaling the data yields worse results in the cross-validation
        # scaler = StandardScaler()
        # x_train = scaler.fit_transform(x_train)
        # x_test = scaler.transform(x_test)

        model.fit(x_train, y_train)
        out_train = model.predict(x_train)
        score_train = concordance_index_censored(y_train["Relapse"], y_train["RFS"], out_train)
        out = model.predict(x_test)
        score = concordance_index_censored(y_test["Relapse"], y_test["RFS"], out)
        cis.append(score[0])
        cis_train.append(score_train[0])
print(f"Train C-Index: {np.mean(cis_train)}")
print(f"Test C-Index: {np.mean(cis)}")


"""scores = make_scores(x_train)
best_train_score = -1
for threshold in np.arange(scores.values.min(), scores.values.max(), (scores.values.max() - scores.values.min())/10):
    x_train_copy = copy.deepcopy(x_train)
    x_test_copy = copy.deepcopy(x_test)
    for col in scores.index:
        if scores[scores.index==col].values[0] > threshold:
            pass
        else:
            del x_train_copy[col]
            del x_test_copy[col]

    del x_train_copy["RFS"]
    del x_train_copy["Relapse"]
    del x_train_copy["Patient ID"]
    del x_test_copy["RFS"]
    del x_test_copy["Relapse"]
    del x_test_copy["Patient ID"]
    train_scores = []
    test_scores = []
    for i in range(10):
        model = FastSurvivalSVM(max_iter=2)
        model.fit(x_train_copy, y_train)
        
        out_train = model.predict(x_train_copy)
        score_train = concordance_index_censored(y_train["Relapse"], y_train["RFS"], out_train)
        out = model.predict(x_test_copy)
        score = concordance_index_censored(y_test["Relapse"], y_test["RFS"], out)
        train_scores.append(score_train[0])
        test_scores.append(score[0])
    
    if np.mean(train_scores)>best_train_score:
        best_score = np.mean(test_scores)
        best_train_score = np.mean(train_scores)
        best_threshold = threshold
    print(f"{threshold}: {np.mean(test_scores)}: {np.mean(train_scores)}")
cis.append(best_score)
print(np.mean(cis))"""
