"""Mask transforms"""

import logging
from abc import ABC

import numpy as np
import SimpleITK as sitk


logger = logging.getLogger(__name__)

class MaskLocalTransform(ABC):
    """Abstract class for local mask transforms. Implements method to act only on part of the mask."""
    def __init__(self):
        pass

    def __call__(self, mask, image):
        pass

    def crop(self, mask, image):
        """Crops the image and mask to the smallest bounding box of the mask

        Args:
            mask (SimpleITK.Image): Mask, will be used to determine the bounding box and cropped. Should be binary
            image (SimpleITK.Image): Image, will be cropped as well.

        Returns:
            tuple(SimpleITK.Image, SimpleITK.Image): Cropped mask, Cropped image
        """
        self.original_size = mask.GetSize()
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(mask)
        if label_shape_filter.GetNumberOfLabels() == 0:
            logger.warning("No foreground found in the mask.")
            return mask, image

        self.bounding_box = label_shape_filter.GetBoundingBox(1) 

        self.crop_image_filter = sitk.CropImageFilter()
        self.crop_image_filter.SetLowerBoundaryCropSize(self.bounding_box[:3])
        self.crop_image_filter.SetLowerBoundaryCropSize(self.bounding_box[:3]+self.bounding_box[3:])
        mask = self.crop_image_filter.Execute(mask)
        image = self.crop_image_filter.Execute(image)
        return mask, image

    def revert_crop(self, mask, cropped_mask):
        paste_filter = sitk.PasteImageFilter()
        paste_filter.SetDestinationIndex(self.bounding_box[:3])
        paste_filter.SetSourceSize(self.bounding_box[3:])
        mask = paste_filter.Execute(mask, cropped_mask)
        return mask

class MaskIdentityTransform:
    def __init__(self):
        pass

    def __call__(self, mask, image):
        return mask


class MaskDilateTransform:
    def __init__(self, dilatation_size):
        self.dilatation_size = dilatation_size

    def __call__(self, mask, image):
        spacing = mask.GetSpacing()
        radius = np.round(np.divide(self.dilatation_size, spacing)).astype(np.uint32).tolist()  # Thank you Mixtral 8x7b
        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetKernelRadius(radius)
        out_mask = dilate_filter.Execute(mask)
        out_mask.CopyInformation(mask)
        return out_mask

class MaskSUVThresholdAbsoluteTransform(MaskLocalTransform):
    def __init__(self, threshold):
        self.threshold = threshold        

    def __call__(self, mask, image):
        # SwigPyObjects, cannot be pickled for multiprocessing
        # Hence we define them in the call, not the initialization
        self.threshold_image_filter = sitk.BinaryThresholdImageFilter()
        self.threshold_image_filter.SetLowerThreshold(self.threshold)
        self.threshold_image_filter.SetOutsideValue(0)
        self.threshold_image_filter.SetInsideValue(1)

        self.multiply_image_filter = sitk.MultiplyImageFilter()

        cropped_mask, image = self.crop(mask, image)
        cropped_mask = sitk.Cast(cropped_mask, sitk.sitkFloat32)

        voi_image = self.multiply_image_filter.Execute(image, cropped_mask)

        cropped_mask = self.threshold_image_filter.Execute(voi_image)

        return self.revert_crop(mask, cropped_mask)


class MaskSUVThresholdRelativeTransform(MaskLocalTransform):
    def __init__(self, threshold_prct):
        self.threshold_prct = threshold_prct

    def __call__(self, mask, image):
        # SwigPyObjects, cannot be pickled for multiprocessing
        # Hence we define them in the call, not the initialization
        self.threshold_image_filter = sitk.BinaryThresholdImageFilter()
        self.threshold_image_filter.SetOutsideValue(0)
        self.threshold_image_filter.SetInsideValue(1)
        self.multiply_image_filter = sitk.MultiplyImageFilter()
        self.max_image_filter = sitk.MinimumMaximumImageFilter()

        cropped_mask, image = self.crop(mask, image)
        cropped_mask = sitk.Cast(cropped_mask, sitk.sitkFloat32)

        voi_image = self.multiply_image_filter.Execute(image, cropped_mask)

        self.max_image_filter.Execute(voi_image)
        max_value = self.max_image_filter.GetMaximum()

        self.threshold_image_filter.SetLowerThreshold(self.threshold_prct*max_value)

        cropped_mask = self.threshold_image_filter.Execute(voi_image)
        return self.revert_crop(mask, cropped_mask)
