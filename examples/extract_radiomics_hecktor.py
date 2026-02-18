import os
import sys
import logging
from tqdm import tqdm
from os.path import join

sys.path.append("..")
from utils.volumes.images import load_image
from utils.volumes.masks import load_mask
from utils.radiomics.extraction import Radiomics_Extractor
from utils.volumes.masks import resample_mask


os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename='logs/hecktor.log', level=logging.INFO)
dataset_root = "/media/zhack/T7/Zach/Data/hecktor2022"

images_dir = join(dataset_root, "imagesTr")
labels_dir = join(dataset_root, "labelsTr")
list_ids = [i[:-7] for i in sorted(os.listdir(labels_dir))]
os.makedirs("../csvs", exist_ok=True)
csv_file = open(join("../csvs", "Hecktor22_Radiomics.csv"), "w", encoding="utf-8")

# Write header
for patient_id in tqdm(list_ids[0:1]):
    csv_file.write("Patient ID")
    ct_image_path = join(images_dir, f"{patient_id}_0001.nii.gz")
    pet_image_path = join(images_dir, f"{patient_id}_0001.nii.gz")
    mask_path = join(labels_dir, f"{patient_id}.nii.gz")
    
    ct_image = load_image(ct_image_path)
    pet_image = load_image(pet_image_path)
    mask_tumor = load_mask(mask_path, label=1)
    mask_lymphnode = load_mask(mask_path, label=2)
    
    ct_tumor_extractor = Radiomics_Extractor(ct_image, mask_tumor)
    ct_node_extractor = Radiomics_Extractor(ct_image, mask_lymphnode)
    pet_tumor_extractor = Radiomics_Extractor(pet_image, mask_tumor)
    pet_node_extractor = Radiomics_Extractor(pet_image, mask_lymphnode)

    vector_ct_tumor = ct_tumor_extractor.get_feature_vector()
    vector_ct_node = ct_node_extractor.get_feature_vector()
    vector_pet_tumor = pet_tumor_extractor.get_feature_vector()
    vector_pet_node = pet_node_extractor.get_feature_vector()
    
    for key, value in vector_ct_tumor.items():
        csv_file.write(f",CT_Tumor_{key}")
    for key, value in vector_ct_node.items():
        csv_file.write(f",CT_Node_{key}")
    for key, value in vector_pet_tumor.items():
        csv_file.write(f",PET_Tumor_{key}")
    for key, value in vector_pet_node.items():
        csv_file.write(f",PET_Node_{key}")
    csv_file.write("\n")

# Write data
for patient_id in tqdm(list_ids):
    csv_file.write(patient_id)
    ct_image_path = join(images_dir, f"{patient_id}_0001.nii.gz")
    pet_image_path = join(images_dir, f"{patient_id}_0001.nii.gz")
    mask_path = join(labels_dir, f"{patient_id}.nii.gz")
    
    ct_image = load_image(ct_image_path)
    pet_image = load_image(pet_image_path)
    mask_tumor = load_mask(mask_path, label=1)
    mask_lymphnode = load_mask(mask_path, label=2)

    # Resample masks to PET image spacing
    mask_tumor = resample_mask(mask_tumor, to=pet_image)
    mask__lymphnode = resample_mask(mask_lymphnode, to=pet_image)
    
    ct_tumor_extractor = Radiomics_Extractor(ct_image, mask_tumor)
    ct_node_extractor = Radiomics_Extractor(ct_image, mask_lymphnode)
    pet_tumor_extractor = Radiomics_Extractor(pet_image, mask_tumor)
    pet_node_extractor = Radiomics_Extractor(pet_image, mask_lymphnode)

    vector_ct_tumor = ct_tumor_extractor.get_feature_vector()
    vector_ct_node = ct_node_extractor.get_feature_vector()
    vector_pet_tumor = pet_tumor_extractor.get_feature_vector()
    vector_pet_node = pet_node_extractor.get_feature_vector()
    
    for key, value in vector_ct_tumor.items():
        csv_file.write(f",{value}")
    for key, value in vector_ct_node.items():
        csv_file.write(f",{value}")
    for key, value in vector_pet_tumor.items():
        csv_file.write(f",{value}")
    for key, value in vector_pet_node.items():
        csv_file.write(f",{value}")
    csv_file.write("\n")
    
csv_file.close()