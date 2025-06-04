import sys
import pandas
import sksurv
import sklearn
import numpy as np
from tqdm import tqdm
from itertools import product
from sksurv.svm import FastSurvivalSVM
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import StratifiedKFold
sys.path.append("../../")
from utils.training.features.selection import get_duplicates, get_best_features



RADIOMICS_PATH = "../../csvs/Hecktor22_AugmentedRadiomics.csv"
ENDPOINTS_PATH = "/media/zhack/Toshiba/drawer/hecktor2022/hecktor2022_endpoint_training.csv"
CLINICAL_DATA="/media/zhack/Toshiba/drawer/hecktor2022/hecktor2022_clinical_info_training.csv"
# Loading radiomics
hecktor_radiomics = pandas.read_csv(RADIOMICS_PATH,
                                    dtype={"Elongation": float, "Flatness": float,
                                           "MajorAxisLength": float, "MinorAxisLength": float,
                                           "LeastAxisLength": float})
# Loading endpoints
endpoints = pandas.read_csv(ENDPOINTS_PATH)
# Loading clinical data
clinical_data = pandas.read_csv(CLINICAL_DATA)

# Removing patients not present in the endpoints csv file (Task 1-only patients)
patient_ids_task2 = endpoints["PatientID"]
hecktor_radiomics = hecktor_radiomics[hecktor_radiomics["Patient ID"].apply(
    lambda x: x.split("_")[0]).isin(patient_ids_task2)]
print(f"Found {len(hecktor_radiomics)} patients with task 2 endpoint")

# Merging the clinical and radiomics data
hecktor_radiomics["Patient Name"] = hecktor_radiomics["Patient ID"].apply(lambda x: x.split("_")[0])
hecktor_radiomics = pandas.merge(hecktor_radiomics, clinical_data, left_on="Patient Name", right_on="PatientID")

# Adapting the clinical data taxonomy
hecktor_radiomics["Gender"] = hecktor_radiomics["Gender"].apply(lambda x: 1 if x =="M" else 2)
hecktor_radiomics["Tobacco"] = hecktor_radiomics["Tobacco"].apply(lambda x: 1 if x ==1 else -1 if x==0 else 0)
hecktor_radiomics["Surgery"] = hecktor_radiomics["Surgery"].apply(lambda x: 1 if x ==1 else -1 if x==0 else 0)
hecktor_radiomics["Chemotherapy"] =hecktor_radiomics["Chemotherapy"].apply(lambda x: 1 if x ==1 else -1 if x==0 else 0)
hecktor_radiomics["Performance status"] =hecktor_radiomics["Performance status"].apply(lambda x: x+1 if x in (0,1,2,3,4) else 0)
hecktor_radiomics["HPV status (0=-, 1=+)"] = hecktor_radiomics["HPV status (0=-, 1=+)"].apply(lambda x: 1 if x ==1 else -1 if x==0 else 0)
del hecktor_radiomics["PatientID"]
#del hecktor_radiomics["CenterID"]

# Merging the endpoints and data
hecktor_radiomics = pandas.merge(hecktor_radiomics, endpoints,
                                 left_on="Patient Name", right_on="PatientID")
del hecktor_radiomics["Patient Name"]
del hecktor_radiomics["PatientID"]
del hecktor_radiomics["Task 1"]
del hecktor_radiomics["Task 2"]

for column in hecktor_radiomics.columns:
    if column.split("_")[-1] in ["Elongation", "Flatness", "LeastAxisLength", "MinorAxisLength", "MajorAxisLength"]:
        hecktor_radiomics[column] = hecktor_radiomics[column].apply(lambda x: np.real(np.complex64(x)))


# Computing the stratified KFold
patient_ids_new = endpoints["PatientID"]
relapse = endpoints["Relapse"]
np.random.seed(1053)
sklearn.random.seed(1053)
kfold = StratifiedKFold(n_splits=5, shuffle=True)
splits = kfold.split(patient_ids_new.unique(), y=relapse)

# Remove duplicates
duplicate_columns = get_duplicates(hecktor_radiomics)
for c in duplicate_columns:
    if c not in ["RFS", "Relapse", "Patient ID"]:
        del hecktor_radiomics[c]
hecktor_radiomics = hecktor_radiomics.fillna(0)

cis = []
cis_train = []
# Performing Cross-Validation
for split in splits:
    # Retrieving the training / testing split patient ids
    train_ids = []
    test_ids = []
    train_indices = split[0]
    test_indices = split[-1]

    # Split train/test
    for id_ in hecktor_radiomics["Patient ID"]:
        for tr_idx in train_indices:
            if id_.startswith(patient_ids_new[tr_idx]):
                train_ids.append(id_)
                break
    for id_ in hecktor_radiomics["Patient ID"]:
        for ts_idx in test_indices:
            if id_.startswith(patient_ids_new[ts_idx]):
                test_ids.append(id_)
                break
    x_train = hecktor_radiomics[hecktor_radiomics["Patient ID"].isin(train_ids)]
    x_test = hecktor_radiomics[hecktor_radiomics["Patient ID"].isin(test_ids)]
    
    # Keep only real data for testing
    x_test = x_test[x_test["Patient ID"].str.endswith("_Identity_Identity")]
    
    
    def renaming_fn(x):
        return x.split("_")[0]
    
    it = ["Identity", "Noise", "Blur"]  # "SimulateLowRes"
    mt = ["Identity", "Dilate4mm", "SUVThresholdRel.4"]
    x_train = x_train[x_train["Patient ID"].str.endswith(tuple(["_".join(i) for i in product(it,mt)]))]

    x_train.loc[:,"Patient ID"] = x_train["Patient ID"].apply(renaming_fn)
    x_test.loc[:,"Patient ID"] = x_test["Patient ID"].apply(renaming_fn)


    
    
    y_train = sksurv.util.Surv.from_dataframe("Relapse", "RFS", x_train)
    y_test = sksurv.util.Surv.from_dataframe("Relapse", "RFS", x_test)


    del x_train["RFS"]
    del x_train["Relapse"]
    del x_train["Patient ID"]
    del x_test["RFS"]
    del x_test["Relapse"]
    del x_test["Patient ID"]
    
    # Drop non-correlated features
    best_features = get_best_features(x_train, y_train, nb_features=250)
    #non_correlated_features = get_noncorrelated_features(x_train, y_train, threshold=.62)
    for feature in x_train.columns:
        if feature not in best_features:
            del x_train[feature]
            del x_test[feature]

    for i in tqdm(range(10)):
        #model = RandomSurvivalForest(n_estimators=20, max_features="sqrt", n_jobs=-1)
        model = FastSurvivalSVM(max_iter=3)
        
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
