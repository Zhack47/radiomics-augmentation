import os
import sys
import multiprocessing
from tqdm import tqdm
from time import sleep, time_ns
from os.path import join
import SimpleITK as sitk
import numpy as np

sys.path.append("../../")
from utils.volumes.images import load_image
from utils.volumes.masks import load_mask, resample_mask
from utils.transforms.masks.mask_transforms import MaskDilateTransform
from utils.transforms.images.image_transforms import ImageNoiseTransform
from utils.radiomics.extraction import Radiomics_Extractor


def make_header(modality_names, mask_names, feature_names):
    header = "Patient_ID"
    for modality_name in modality_names:
        for mask_name in mask_names:
            for feature_name in feature_names:
                header += f",{modality_name}_{mask_name}_{feature_name}"
    return header + "\n"


def augment_and_extract(patient, patient_id, image_aug, mask_aug):
    ret = f"{patient_id}" \
          f"_{image_aug[0]}" \
          f"_{mask_aug[0]}"
    for image in patient["Images"]:

        sitk_image = load_image(image)
        transformed_image = image_aug[1](sitk_image)

        for mask, label in patient["Masks"]:

            sitk_mask = load_mask(mask, label)
            sitk_mask = resample_mask(sitk_mask, sitk_image)
            transformed_mask = mask_aug[1](sitk_mask, sitk_image)

            extractor = Radiomics_Extractor(transformed_image, transformed_mask)

            del sitk_mask
            del transformed_mask

            feature_vector = extractor.get_feature_vector()
            features_str = ",".join(list(map(str, feature_vector.values())))
            ret += f",{features_str}"

        del transformed_image
        del sitk_image
    return ret + "\n"


def augment_and_extract_with_multiprocessing(patients,
                                             list_image_augmentations,
                                             list_mask_augmentations,
                                             csv_file, num_processes=3):
    task_items = []
    for patient in patients:
        for image_aug in list_image_augmentations:
            for mask_aug in list_mask_augmentations:
                task_items.append((patients[patient], patient,
                                   image_aug, mask_aug))

    r = []
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        with tqdm(desc="Extracting Radiomics", total=len(task_items)) as pbar:
            workers = [j for j in p._pool]

            for task in task_items:
                r.append(p.starmap_async(augment_and_extract, [task]))
            remaining = list(range(len(task_items)))
            while len(remaining) > 0:
                all_alive = all([j.is_alive() for j in workers])
                if not all_alive:
                    raise RuntimeError('A worker died\n'
                                        'This could be because of '
                                        'an error (look for an error message) or because it was killed '
                                        'by your OS due to running out of RAM. If you don\'t see '
                                        'an error message, out of RAM is likely the problem. In that case '
                                        'reducing the number of workers might help')
                done = [i for i in remaining if r[i].ready()]

                for i in done:
                    row = r[i].get()[0]
                    csv_file.write(row)

                    pbar.update()

                remaining = [i for i in remaining if i not in done]
                sleep(0.1)


if __name__ == "__main__":
    root = "/media/zhack/T7/Zach/Data/hecktor2025_small/"
    patient_ids = [i for i in os.listdir(root) if not i.endswith(".csv")]

    # modalities = {"CT": "__CT", "RTDOSE": "__RTDOSE", "PT": "__PT"}
    modalities = {"CT": "__CT", "PT": "__PT"}
    labels = [1, 2]

    im_augs = [("Noise0.1", ImageNoiseTransform(.1)),
               ("Noise0.2", ImageNoiseTransform(.2))]
    masks_augs = [("Dilate4", MaskDilateTransform(4)),
                  ("Dilate2", MaskDilateTransform(2))]

    patients = {}
    for pat_id in patient_ids:
        patients[pat_id] = {}
        patients[pat_id]["Images"] = []
        patients[pat_id]["Masks"] = []
        for modality in modalities:
            patients[pat_id]["Images"].append(join(root,
                                                   pat_id,
                                                   f"{pat_id}{modalities[modality]}.nii.gz"))
        for label in labels:
            patients[pat_id]["Masks"].append((join(root,
                                                   pat_id,
                                                   f"{pat_id}.nii.gz"), label))

    time_0 = time_ns()
    csv_file_ = open("../../csvs/test_mp.csv", "w")
    mask_names = ["tumor", "nodes"]

    toy_array = np.zeros((64, 64, 64),dtype=np.uint16)
    toy_array[16:48] += 1
    toy_image = sitk.GetImageFromArray(toy_array)
    extractor = Radiomics_Extractor(toy_image, toy_image)
    feature_names = list(extractor.get_feature_vector().keys())

    header = make_header(list(modalities.keys()), mask_names, feature_names)
    csv_file_.write(header)
    augment_and_extract_with_multiprocessing(patients, im_augs, masks_augs,
                                             csv_file_, num_processes=2)

    print((time_ns() - time_0)/1e9)
    print("__________________")
    i = 0
    time_1 = time_ns()
    for pat_id in tqdm(patient_ids):
        for image in patients[pat_id]["Images"]:
            sitk_image_ = load_image(image)
            for mask in patients[pat_id]["Masks"]:
                sitk_mask_ = load_mask(mask[0], mask[1])
                sitk_mask_ = resample_mask(sitk_mask_, sitk_image_)
                for image_aug in im_augs:
                    sitk_image_ = image_aug[1](sitk_image_)
                    for mask_aug in masks_augs:
                        sitk_mask_ = mask_aug[1](sitk_mask_, sitk_image_)
                        extractor = Radiomics_Extractor(sitk_image_, sitk_mask_)
                        fv = extractor.get_feature_vector()
                        i += 1
    print((time_ns() - time_1)/1e9)
    print(f"Did {i} Iterations")
    print("__________________")
