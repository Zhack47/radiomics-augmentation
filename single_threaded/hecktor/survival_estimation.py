import sys
import pandas
import sksurv
import sklearn
import numpy as np
from tqdm import tqdm
from sksurv.svm import FastSurvivalSVM
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import StratifiedKFold
sys.path.append("../../")
from utils.training.features.selection import get_duplicates, get_best_features


# Defining paths
RADIOMICS_PATH = "../../csvs/Hecktor22_Radiomics.csv"
ENDPOINTS_PATH = "/media/zhack/Toshiba/drawer/hecktor2022/hecktor2022_endpoint_training.csv"
CLINICAL_PATH = "/media/zhack/Toshiba/drawer/hecktor2022/hecktor2022_clinical_info_training.csv"

# Loading radiomics
hecktor_radiomics = pandas.read_csv(RADIOMICS_PATH)
# Loading endpoints
endpoints = pandas.read_csv(ENDPOINTS_PATH)
# Loading clinical data
clinical_data = pandas.read_csv(CLINICAL_PATH)

# Removing patients not present in the endpoints csv file (Task 1-only patients)
patient_ids_task2 = endpoints["PatientID"]
hecktor_radiomics = hecktor_radiomics[hecktor_radiomics["Patient ID"].isin(patient_ids_task2)]
print(f"Found {len(hecktor_radiomics)} patients with task 2 endpoint")

# Merging the clinical and radiomics data
hecktor_radiomics = pandas.merge(hecktor_radiomics, clinical_data,
                                 left_on="Patient ID", right_on="PatientID")
del hecktor_radiomics["PatientID"]
del hecktor_radiomics["Task 1"]
del hecktor_radiomics["Task 2"]
# del hecktor_radiomics["CenterID"]  # Maybe remove the CenterID variable

# Adapting the clinical data taxonomy
hecktor_radiomics["Gender"] = hecktor_radiomics["Gender"].apply(lambda x: 1 if x =="M" else 2)
hecktor_radiomics["Tobacco"] = hecktor_radiomics["Tobacco"].apply(lambda x: 1 if x ==1 else -1 if x==0 else 0)
hecktor_radiomics["Surgery"] = hecktor_radiomics["Surgery"].apply(lambda x: 1 if x ==1 else -1 if x==0 else 0)
hecktor_radiomics["Chemotherapy"] =hecktor_radiomics["Chemotherapy"].apply(lambda x: 1 if x ==1 else -1 if x==0 else 0)
hecktor_radiomics["HPV status (0=-, 1=+)"] = hecktor_radiomics["HPV status (0=-, 1=+)"].apply(lambda x: 1 if x ==1 else -1 if x==0 else 0)
hecktor_radiomics["Performance status"] =hecktor_radiomics["Performance status"].apply(lambda x: x+1 if x in (0,1,2,3,4) else 0)

# Merging the endpoints and data
hecktor_radiomics = pandas.merge(hecktor_radiomics,
                                 endpoints,left_on="Patient ID", right_on="PatientID")
del hecktor_radiomics["PatientID"]

# Computing the stratified KFold
patient_ids_new = hecktor_radiomics["Patient ID"]
relapse = hecktor_radiomics["Relapse"]
np.random.seed(1053)
sklearn.random.seed(1053)
kfold = StratifiedKFold(n_splits=5, shuffle=True)
splits = kfold.split(patient_ids_new, y=relapse)

# Main loop
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

    # Split the data in train/test
    x_train = hecktor_radiomics[hecktor_radiomics["Patient ID"].isin(train_ids)]
    x_test = hecktor_radiomics[hecktor_radiomics["Patient ID"].isin(test_ids)]
    y_train = sksurv.util.Surv.from_dataframe("Relapse", "RFS", x_train)
    y_test = sksurv.util.Surv.from_dataframe("Relapse", "RFS", x_test)

    x_train = x_train.fillna(0)
    x_test = x_test.fillna(0)

    # Remove duplicates
    duplicate_columns = get_duplicates(x_train)
    for c in duplicate_columns:
        if c not in ["RFS", "Relapse", "Patient ID"]:
            del x_train[c]
            del x_test[c]

    # Remove target and name variables
    del x_train["RFS"]
    del x_train["Relapse"]
    del x_train["Patient ID"]
    del x_test["RFS"]
    del x_test["Relapse"]
    del x_test["Patient ID"]
    
    # Drop non-correlated features, either by correlation (c-index) or by number (n best)
    # non_correlated_features = get_noncorrelated_features(x_train, y_train, threshold=.52)
    best_features = get_best_features(x_train, y_train, nb_features=40)
    for feature in x_train.columns:
        if feature not in best_features:
            del x_train[feature]
            del x_test[feature]

    # We run 10 times our CV computation
    for i in tqdm(range(10)):
        #model = RandomSurvivalForest(n_estimators=20, max_features="sqrt", n_jobs=-1)
        model = FastSurvivalSVM(max_iter=3)
        
        # Scaling the data yields worse results in the cross-validation
        # scaler = StandardScaler()
        # x_train = scaler.fit_transform(x_train)
        # x_test = scaler.transform(x_test)

        model.fit(x_train, y_train)
        out_train = model.predict(x_train)  # Train metric for model selection
        score_train = concordance_index_censored(y_train["Relapse"], y_train["RFS"], out_train)
        out = model.predict(x_test)
        score = concordance_index_censored(y_test["Relapse"], y_test["RFS"], out)
        cis.append(score[0])
        cis_train.append(score_train[0])

print(f"Train C-Index: {np.mean(cis_train)}")
print(f"Test C-Index: {np.mean(cis)}")
