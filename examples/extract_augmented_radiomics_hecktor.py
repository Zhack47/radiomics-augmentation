import os
from os.path import join
import sys
import logging
from tqdm import tqdm

sys.path.append("..")
from utils.volumes.images import load_image
from utils.volumes.masks import load_mask
from utils.radiomics.extraction import Radiomics_Extractor
from utils.volumes.masks import resample_mask
from utils.transforms.images.image_transforms import ImageBlurTransform, ImageGammaTransform,\
    ImageContrastShiftTransform, ImageIdentityTransform, ImageMultiplicativeBrightnessTransform,\
    ImageNoiseTransform, ImageSimulateLowResTransform
from utils.transforms.masks.mask_transforms import MaskDilateTransform, MaskIdentityTransform,\
    MaskSUVThresholdAbsoluteTransform, MaskSUVThresholdRelativeTransform


os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename='logs/hecktor.log', level=logging.INFO)

# The root is the 'hecktor2022' folder 
# (if no modification was made after downloading it from the challenge website)
DATASET_ROOT = input("Please input the root location of the hecktor2022 dataset:")


IMAGE_AUGMENTATIONS = {"Identity":ImageIdentityTransform(), "Blur": ImageBlurTransform(4),
                       "Gamma.9": ImageGammaTransform(.9),
                       "ContrastShift.9": ImageContrastShiftTransform(.9),
                       "ContrastShift1.1": ImageContrastShiftTransform(1.1),
                       "MultiplicativeBrightness.8": ImageMultiplicativeBrightnessTransform(.8),
                       "MultiplicativeBrightness1.2": ImageMultiplicativeBrightnessTransform(1.2),
                       "Noise": ImageNoiseTransform(.1),
                       "SimulateLowRes": ImageSimulateLowResTransform(.8)
                       }

MASK_AUGMENTATIONS = {"Identity": MaskIdentityTransform(),
                      "Dilate4mm": MaskDilateTransform(4),
                      "SUVThresholdRel.4": MaskSUVThresholdRelativeTransform(.4),
                      "SUVAbsolute2.5": MaskSUVThresholdAbsoluteTransform(2.5),
                      "SUVAbsolute4": MaskSUVThresholdAbsoluteTransform(4)
                       }


# Set the paths to images and label masks
images_dir = join(DATASET_ROOT, "imagesTr")
labels_dir = join(DATASET_ROOT, "labelsTr")

# Retrieving the list of patient IDs
list_ids = [i[:-7] for i in sorted(os.listdir(labels_dir))]

# Creating the output CSV file
os.makedirs("../csvs", exist_ok=True)
csv_file = open(join("..", "csvs", "Hecktor22_AugmentedRadiomics.csv"), "w", encoding="utf-8")

# Writing the header of the CSV file
# This is done by performing a dry-run on the first patient,
# in order to obtain the names of the radiomic features
patient_id = list_ids[0]
csv_file.write("Patient ID")
ct_image_path = join(images_dir, f"{patient_id}_0001.nii.gz")
pet_image_path = join(images_dir, f"{patient_id}_0001.nii.gz")
mask_path = join(labels_dir, f"{patient_id}.nii.gz")

ct_image = load_image(ct_image_path)
pet_image = load_image(pet_image_path)
mask_tumor = load_mask(mask_path, label=1)
mask_lymphnode = load_mask(mask_path, label=2)

# Initializing Radiomics_Extractor objects for each modality/label pair
ct_tumor_extractor = Radiomics_Extractor(ct_image, mask_tumor)
ct_node_extractor = Radiomics_Extractor(ct_image, mask_lymphnode)
pet_tumor_extractor = Radiomics_Extractor(pet_image, mask_tumor)
pet_node_extractor = Radiomics_Extractor(pet_image, mask_lymphnode)

# Retrieving the feature vector corresponding to each modality/label pair
vector_ct_tumor = ct_tumor_extractor.get_feature_vector()
vector_ct_node = ct_node_extractor.get_feature_vector()
vector_pet_tumor = pet_tumor_extractor.get_feature_vector()
vector_pet_node = pet_node_extractor.get_feature_vector()

# Writing the feature names in the header of the output CSV file
# Each modality/label is treated in its own for-loop, but they end up on the same row
for key, value in vector_ct_tumor.items():
    csv_file.write(f",CT_Tumor_{key}")
for key, value in vector_ct_node.items():
    csv_file.write(f",CT_Node_{key}")
for key, value in vector_pet_tumor.items():
    csv_file.write(f",PET_Tumor_{key}")
for key, value in vector_pet_node.items():
    csv_file.write(f",PET_Node_{key}")
csv_file.write("\n")

# Writing radiomic data to the CSV file
# We process one patient at a time, applying all mask/image transforms pairs
# before moving to the next one
for patient_id in tqdm(list_ids):
    ct_image_path = join(images_dir, f"{patient_id}_0001.nii.gz")
    pet_image_path = join(images_dir, f"{patient_id}_0001.nii.gz")
    mask_path = join(labels_dir, f"{patient_id}.nii.gz")

    # Loading modalities images and label masks    
    ct_image = load_image(ct_image_path)
    pet_image = load_image(pet_image_path)
    mask_tumor = load_mask(mask_path, label=1)
    mask_lymphnode = load_mask(mask_path, label=2)

    # Resample masks to PET image spacing
    mask_tumor = resample_mask(mask_tumor, to=pet_image)
    mask_lymphnode = resample_mask(mask_lymphnode, to=pet_image)

    for image_augmentation_name, image_augmentation in IMAGE_AUGMENTATIONS.items():
        # Applying the current image transform to the images 
        augmented_ct_image = image_augmentation(ct_image)
        augmented_pet_image = image_augmentation(pet_image)

        for mask_augmentation_name, mask_augmentation in MASK_AUGMENTATIONS.items():
            # Writing the name of the augmented sample
            csv_file.write(f"{patient_id}_{image_augmentation_name}_{mask_augmentation_name}")

            # Applying the current mask transform to the masks
            augmented_mask_tumor = mask_augmentation(mask_tumor, pet_image)
            augmented_mask_lymphnode = mask_augmentation(mask_lymphnode, pet_image)
            
            # Initializing Radiomics_Extractor objects for each modality/label pair
            ct_tumor_extractor = Radiomics_Extractor(augmented_ct_image, augmented_mask_tumor)
            ct_node_extractor = Radiomics_Extractor(augmented_ct_image, augmented_mask_lymphnode)
            pet_tumor_extractor = Radiomics_Extractor(augmented_pet_image, augmented_mask_tumor)
            pet_node_extractor = Radiomics_Extractor(augmented_pet_image, augmented_mask_lymphnode)

            # Retrieving the feature vector corresponding to each modality/label pair
            vector_ct_tumor = ct_tumor_extractor.get_feature_vector()
            vector_ct_node = ct_node_extractor.get_feature_vector()
            vector_pet_tumor = pet_tumor_extractor.get_feature_vector()
            vector_pet_node = pet_node_extractor.get_feature_vector()
            
            # Write the radiomic feature values to the CSV file
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
