## Inspired from https://github.com/Lrebaud/ICARE/blob/main/notebook/reproducing_HECKTOR2022.ipynb


import numpy as np
import warnings
import pandas as pd
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sklearn.model_selection import StratifiedKFold
from sksurv.svm import FastSurvivalSVM
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from icare.survival import BaggedIcareSurvival
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

import sys
sys.path.append("..")
from utils.training.features.selection import get_duplicates, f_uci
from tqdm import tqdm


warnings.filterwarnings("ignore")


if len(sys.argv)!=3:
    print("Usage python3 Hecktor22.py [model name] [Augmented:(True/False)]")
    exit()

if sys.argv[2]=="False":
    augmented = False
elif sys.argv[2]=="True":
    augmented=True
else:
    raise ValueError("Augmented must be True or False")


# Select the model among the ones available
model_name = sys.argv[1]
models = {"icare10" : BaggedIcareSurvival(n_estimators=10, parameters_sets=None, aggregation_method='median', n_jobs=-1),
          "icare100": BaggedIcareSurvival(n_estimators=100, parameters_sets=None, aggregation_method='median', n_jobs=-1),
          "FS-SVM"  : FastSurvivalSVM(),
          "cox"     : CoxnetSurvivalAnalysis(),
          "rsf100"  : RandomSurvivalForest(n_estimators=100, n_jobs=-1),
          "rsf10"   : RandomSurvivalForest(n_estimators=10),
          "gb10"    : GradientBoostingSurvivalAnalysis(n_estimators=10),
          "gb100"   : GradientBoostingSurvivalAnalysis(n_estimators=100)
          }

if model_name not in models.keys():
    raise ValueError(f"Model {model_name} not found. Please choose among {models.keys()}")

n_repeats = 10

# Opening a CSV file to store metric values
if augmented:
    csv_file = open(f"../csvs/Performance/Perf_Hecktor_augmented_{model_name}.csv", "w")
else:
    csv_file = open(f"../csvs/Performance/Perf_Hecktor_base_{model_name}.csv", "w")

csv_file.write("nb_variables,CI_train,CI_test,cdAUC_train,cdAUC_test,ConfInt\n")

# opaning the original/augmented radiomics
# We have noticed issues with some features (see below)
# While we used a defined type while loading the CSV with pandas,
# we also built a type check/casting below
augmented_radiomics = pd.read_csv("../csvs/Hecktor22_AugmentedRadiomics_mp.csv",
                                    dtype={"Elongation": float, "Flatness": float,
                                        "MajorAxisLength": float, "MinorAxisLength": float,
                                        "LeastAxisLength": float})

# Type check/casting
for column in augmented_radiomics.columns:
    if column.split("_")[-1] in ["Elongation", "Flatness", "LeastAxisLength", "MinorAxisLength", "MajorAxisLength"]:
        augmented_radiomics[column] = augmented_radiomics[column].apply(lambda x: np.real(np.complex64(x)))

#Retrieving original Patient ID, set in the column Patient Name
augmented_radiomics["Patient Name"] = augmented_radiomics["Patient ID"].apply(lambda x: x.split("_")[0])

# Loading original radiomics
normal_radiomics = pd.read_csv("../csvs/Hecktor22_Radiomics.csv", index_col="Patient ID")
normal_radiomics["Patient Name"] = normal_radiomics.index

# Loading clinical data
if augmented:
    endpoints = pd.read_csv("../csvs/hecktor2022_endpoint_training.csv")  # Maybe try using index again
    clinical = pd.read_csv("../csvs/hecktor2022_clinical_info_training.csv")  # Maybe try using index again
else:
    endpoints = pd.read_csv("../csvs/hecktor2022_endpoint_training.csv", index_col="PatientID")
    clinical = pd.read_csv("../csvs/hecktor2022_clinical_info_training.csv", index_col="PatientID")

# Encoding clinical data
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


# Maybe we should normalize the whole df,  or at least all samples coming from a same patient
# before computing the distance
if augmented:
    '''
    # This is future work for sample selection
    def get_ridiculous_augments(df: pd.DataFrame, thresh):
        remove = []
        for pname in tqdm(df["Patient Name"].unique()):
            true = copy.deepcopy(df[df["Patient ID"]==f"{pname}_Identity_Identity"])
            augs = copy.deepcopy(df[(df["Patient Name"]==pname) & (df["Patient ID"]!=f"{pname}_Identity_Identity")])
            del true["Patient ID"]
            del true["Patient Name"]
            for aug_id in augs["Patient ID"]:
                aug = copy.deepcopy(augs[augs["Patient ID"]==aug_id])
                del aug["Patient Name"]
                del aug["Patient ID"]
                aug = aug.fillna(0)
                true = true.fillna(0)
                aug_values = aug.values[0]
                true_values = true.values[0]
                euc_distance = distance.euclidean(aug_values, true_values, abs(1 / (true_values+1e-3))) / len(aug_values)
                if euc_distance < thresh:
                    remove.append(aug_id)
        return remove
    print(df_train.shape)
    print("Removing augmentations too far from reality")
    to_remove = get_ridiculous_augments(df_train,np.inf)
    df_train = df_train[~df_train["Patient ID"].isin(to_remove)]        
    '''
    # Merge the radiomics with clinical data and endpoints
    #print(df_train.shape)
    df_train = pd.merge(df_train, clinical, left_on="Patient Name", right_on="PatientID")
    #print(df_train.shape)
    df_train = pd.merge(df_train, endpoints, left_on="Patient Name", right_on="PatientID")
    #print(df_train.shape)
    

    df_train = df_train.set_index("Patient ID")
    #del df_train["Patient Name"]
    #del df_train["Patient ID"]
    del df_train["PatientID_y"]
    del df_train["PatientID_x"]
else:
    # Merge the radiomics with clinical data and endpoints
    df_train = pd.merge(df_train, clinical, left_index=True, right_index=True)
    df_train = pd.merge(df_train, endpoints, left_index=True, right_index=True)

# Build the survival array from event and time data
y_train = Surv.from_arrays(event=df_train['Relapse'].values,
                           time=df_train['RFS'].values)

# Initialize the 5-folds cross-validation
kfold = StratifiedKFold(5, random_state=np.random.randint(0, 1e9), shuffle=True)

# Replacing any stray NaNs with zeroes
df_train = df_train.fillna(0)

# Getting and removing duplicate columns
# We used normal_radiomics to remove the same columns regardless of whether
# augmentations were used or not
duplicate_columns = get_duplicates(normal_radiomics)
for c in duplicate_columns:
    if c not in ["RFS", "Relapse", "Patient ID"]:
        del df_train[c]
print(df_train.shape)


# Gather the censoring information to stratify the 5-folds CV with censoring
ids = np.unique(df_train["Patient Name"].values)
if augmented:
    censored = [df_train[df_train.index==f"{x}_Identity_Identity"]["Relapse"] for x in ids]
else:
    censored = df_train["Relapse"]

# The sklearn selectors will be cached to avoid having to recompute
# feature selection at each iteration
selectors = [None]*5

# Let's iterate over all the possible numbers of variables
for thr in tqdm(range(1,df_train.values.shape[1],1)):
    ci_avg_test = 0.
    ci_avg_train = 0.
    cdauc_avg_test=0.
    cdauc_avg_train=0.
    confint_avg_test = 0.
    
    # Splits are made usingg original patient IDs ONLY
    for split_nb, (tr_ids, ts_ids) in enumerate(kfold.split(ids, censored)):
        # Recuperating the train/test IDs
        train_ids = ids[tr_ids]
        test_ids = ids[ts_ids]
        
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
        
        # If the selector was not set for this fold, we put it in the list defined above
        if selectors[split_nb] is None:
            # The selector is built and fitted to the data, but does not select the variables yet at this point
            # This allows us to compute UCI scores only once per fold
            print(f"Initializing selector for fold {split_nb}")
            selector = SelectKBest(score_func=f_uci, k=thr) 
            X_train_selected = selector.fit(X_train_local.values, Y_train_local)
            selectors[split_nb] =selector
        
        # The ._get_support_mask() function uses k and is called by .transform() to select variables
        # We set k and perform selection on the pre-computed UCI scores
        selectors[split_nb].set_params(k=thr)
        #X_train_selected = selectors[split_nb].transform(X_train_local)
        #X_test_selected = selectors[split_nb].transform(X_test_local)

        selected_features = X_train_local.columns[selectors[split_nb].get_support()]
        f_scores = selectors[split_nb].scores_[selectors[split_nb].get_support()]

        X_train_local = X_train_local[selected_features]
        X_test_local = X_test_local[selected_features]
        
        # Using StandardScaler to normalize the data
        scaler = StandardScaler()
        X_train_local_np = scaler.fit_transform(X_train_local)
        X_test_local_np = scaler.transform(X_test_local)

        citr_loc = []
        cits_loc = []
        cdatr_loc = []
        cdats_loc = []
        for i in range(n_repeats):
            model = models[model_name]

            # Fit the model on the original/augmented data
            model.fit(X_train_local, Y_train_local)
            train_pred = model.predict(X_train_local)
            test_pred = model.predict(X_test_local)

            # C-Index
            ci_train = concordance_index_censored(Y_train_local["event"], Y_train_local["time"], train_pred)
            ci_test = concordance_index_censored(Y_test_local["event"], Y_test_local["time"], test_pred)
            
            # cdAUC 
            # Not thoroughly researched, DISMISS IT PLEASE
            time_points_train = np.arange(Y_train_local["time"][np.argpartition(Y_train_local["time"],5)[5]],
                                                            Y_train_local["time"][np.argpartition(Y_train_local["time"],-5)[-5]], 50)
            time_points_test = np.arange(Y_test_local["time"][np.argpartition(Y_test_local["time"],5)[5]],
                                                            Y_test_local["time"][np.argpartition(Y_test_local["time"],-5)[-5]], 50)
            cdauc_train = cumulative_dynamic_auc(Y_train_local, Y_train_local, train_pred, times=time_points_train)[1]
            cdauc_test = cumulative_dynamic_auc(Y_train_local, Y_test_local, test_pred, times=time_points_test)[1]
            citr_loc.append(ci_train[0])
            cits_loc.append(ci_test[0])
            cdatr_loc.append(cdauc_train)
            cdats_loc.append(cdauc_test)

        ci_avg_train+=np.mean(citr_loc)
        ci_avg_test+=np.mean(cits_loc)
        cdauc_avg_train+=np.mean(cdatr_loc)
        cdauc_avg_test+=np.mean(cdats_loc)
        confint_avg_test += 1.96 * (np.nanstd(cits_loc) / np.sqrt(10))
    print(f"Average train CI:{ci_avg_train/5}")
    print(f"Average test CI:{ci_avg_test/5}")
    print(f"Average train cdAUC:{cdauc_avg_train/5}")
    print(f"Average test cdAUC:{cdauc_avg_test/5}")
    print(f"Average test CI ConfInt:{confint_avg_test/5}")
    csv_file.write(f"{thr},{ci_avg_train/5},{ci_avg_test/5},{cdauc_avg_train/5},{cdauc_avg_test/5},{confint_avg_test/5}\n")
