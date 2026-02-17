import os
from os.path import join
import sys
sys.path.append("../../")
from utils.transforms.images.image_transforms import ImageBlurTransform, ImageContrastShiftTransform, ImageGammaTransform, ImageNoiseTransform, ImageMultiplicativeBrightnessTransform, ImageSimulateLowResTransform
from utils.transforms.masks.mask_transforms import MaskDilateTransform, MaskSUVThresholdAbsoluteTransform, MaskSUVThresholdRelativeTransform
from utils.volumes.masks import resample_mask
import SimpleITK as sitk

patient_path = "/media/zhack/T7/Zach/Data/hecktor2022/imagesTr/CHUM-001_0000.nii.gz"
pet_image_path = "/media/zhack/T7/Zach/Data/hecktor2022/imagesTr/CHUM-001_0001.nii.gz"
mask_path = "/media/zhack/T7/Zach/Data/hecktor2022/labelsTr/CHUM-001.nii.gz"
output_path = "/media/zhack/T7/Zach/transformed_images"
os.makedirs(output_path, exist_ok=True)
'''image = sitk.ReadImage(patient_path)
image_bt = ImageBlurTransform(4)(image)
sitk.WriteImage(image_bt, join(output_path, "CHUM-001_0000_Blur.nii.gz"))
image_bt = ImageContrastShiftTransform(1.2)(image)
sitk.WriteImage(image_bt, join(output_path, "CHUM-001_0000_CS.nii.gz"))
image_bt = ImageGammaTransform(.7)(image)
sitk.WriteImage(image_bt, join(output_path, "CHUM-001_0000_Gamma.nii.gz"))
image_bt = ImageSimulateLowResTransform(.4)(image)
sitk.WriteImage(image_bt, join(output_path, "CHUM-001_0000_SLR.nii.gz"))
image_bt = ImageMultiplicativeBrightnessTransform(.6)(image)
sitk.WriteImage(image_bt, join(output_path, "CHUM-001_0000_MB.nii.gz"))
image_bt = ImageNoiseTransform(420)(image)
sitk.WriteImage(image_bt, join(output_path, "CHUM-001_0000_Noise.nii.gz"))
'''

mask = sitk.ReadImage(mask_path)
pet_image  = sitk.ReadImage(pet_image_path)
mask = resample_mask(mask, to=pet_image)
sitk.WriteImage(mask, join(output_path, "CHUM-001_resampled_mask.nii.gz"))
mask_bt = MaskDilateTransform(4)(mask, pet_image)
sitk.WriteImage(mask_bt, join(output_path, "CHUM-001_Dilate.nii.gz"))
mask_bt = MaskSUVThresholdRelativeTransform(.4)(mask, pet_image)
sitk.WriteImage(mask_bt, join(output_path, "CHUM-001_Rel.nii.gz"))
mask_bt = MaskSUVThresholdAbsoluteTransform(2.5)(mask, pet_image)
sitk.WriteImage(mask_bt, join(output_path, "CHUM-001_Abs.nii.gz"))
