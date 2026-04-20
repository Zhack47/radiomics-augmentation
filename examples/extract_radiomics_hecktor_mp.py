import os
import sys
import logging
from os.path import join

import numpy as np
import SimpleITK as sitk

sys.path.append("..")
from utils.radiomics.extraction import Radiomics_Extractor
from utils.transforms.masks.mask_transforms import MaskIdentityTransform
from utils.transforms.images.image_transforms import ImageIdentityTransform
from utils.radiomics.extraction_mp import make_header, augment_and_extract_with_multiprocessing


os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename='logs/hecktor.log', level=logging.INFO)

# The root is the 'hecktor2022' folder
# (if no modification was made after downloading it from the challenge website)
DATASET_ROOT = input("Please input the root location of the hecktor2022 dataset:")


images_dir = join(DATASET_ROOT, "imagesTr")
labels_dir = join(DATASET_ROOT, "labelsTr")

modalities = {"CT": "_0000", "PT": "_0001"}
mask_names = ["tumor", "nodes"]
labels = [1, 2]

im_augs = [("Original", ImageIdentityTransform())]
masks_augs = [("Original", MaskIdentityTransform())]

# Retrieving the list of patient IDs
patient_ids = [i for i in os.listdir(labels_dir) if not i.endswith(".csv")]

# Building the dictionary containing the information
# (images paths, mask paths and labels) for each patient
patients = {}
for pat_id in patient_ids:
    patients[pat_id] = {}
    patients[pat_id]["Images"] = []
    patients[pat_id]["Masks"] = []
    for modality in modalities:
        patients[pat_id]["Images"].append(join(images_dir,
                                               f"{pat_id}{modalities[modality]}.nii.gz"))
    for label in labels:
        patients[pat_id]["Masks"].append((join(labels_dir,
                                               f"{pat_id}.nii.gz"), label))

# Creating the output CSV file
os.makedirs("../csvs", exist_ok=True)
csv_file = open(join("../csvs", "Hecktor22_Radiomics.csv"), "w", encoding="utf-8")

# Writing the header of the CSV file
# This is done by performing a dry-run on toy images and masks
# in order to obtain the names of the radiomic features
toy_array = np.zeros((64, 64, 64), dtype=np.uint16)
toy_array[16:48] += 1
toy_image = sitk.GetImageFromArray(toy_array)
extractor = Radiomics_Extractor(toy_image, toy_image)
feature_names = list(extractor.get_feature_vector().keys())

header = make_header(list(modalities.keys()), mask_names, feature_names)
csv_file.write(header)
augment_and_extract_with_multiprocessing(patients, im_augs, masks_augs,
                                         csv_file, num_processes=16)
csv_file.close()
