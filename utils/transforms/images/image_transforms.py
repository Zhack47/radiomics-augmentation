import logging
import torchio as tio
import SimpleITK as sitk


logger = logging.getLogger(__name__)

class ImageIdentityTransform:
    def __init__(self):
        pass
    def __call__(self, image):
        return image


class ImageBlurTransform:
    def __init__(self, std):
        self.std = std
        self.blur_transform = tio.RandomBlur(std=std)
    
    def __call__(self, image):
        return self.blur_transform(image)


class ImageNoiseTransform:
    def __init__(self, std):
        self.std = std
        self.noise_transform = tio.RandomNoise(std=std)
    
    def __call__(self, image):
        return self.noise_transform(image)


class ImageMultiplicativeBrightnessTransform:
    def __init__(self, multiplier):
        self.multiplier = multiplier
    
    def __call__(self, image):
        # Keep the original pixel type
        pixel_id = image.GetPixelID()
        
        # Cast if needed
        if pixel_id not in [sitk.sitkFloat32, sitk.sitkFloat64]:
            sitk.Cast(image, sitk.sitkFloat32)
        
        out_image = self.multiplier * image
        
        # Round and cast back if needed
        if pixel_id in [sitk.sitkUInt8, sitk.sitkUInt16, sitk.sitkUInt32, sitk.sitkUInt64,
                        sitk.sitkInt8, sitk.sitkInt16, sitk.sitkInt32, sitk.sitkInt64]:
            round_image_filter = sitk.RoundImageFilter()
            out_image = round_image_filter.Execute(out_image)
        if pixel_id not in [sitk.sitkFloat32, sitk.sitkFloat64]:
            out_image = sitk.Cast(out_image, image.GetPixelID())
        return out_image


class ImageGammaTransform:
    def __init__(self, gamma):
        self.gamma_transform = tio.Gamma(gamma)
    
    def __call__(self, image):
        return self.gamma_transform(image)


class ImageSimulateLowResTransform:
    def __init__(self, factor, orders=(sitk.sitkNearestNeighbor, sitk.sitkBSpline)):
        self.factor = factor
        self.order_downsampling, self.order_upsampling = orders
    
    def __call__(self, image):
        # SwigPyObjects, cannot be pickled for multiprocessing
        # Hence we define them in the call, not the initialization
        self.downsample_image_filter = sitk.ResampleImageFilter()
        self.upsample_image_filter = sitk.ResampleImageFilter()

        # Set output spaces
        old_origin = image.GetOrigin()
        old_direction = image.GetDirection()
        old_spacing = image.GetSpacing()
        old_size = image.GetSize()

        down_spacing = [i / self.factor for i in old_spacing]
        down_size = [int(round(i*self.factor)) for i in old_size]
        self.downsample_image_filter.SetOutputSpacing(down_spacing)
        self.downsample_image_filter.SetOutputOrigin(old_origin)
        self.downsample_image_filter.SetOutputDirection(old_direction)
        self.downsample_image_filter.SetSize(down_size)
        self.upsample_image_filter.SetInterpolator(self.order_downsampling)

        self.upsample_image_filter.SetOutputOrigin(old_origin)
        self.upsample_image_filter.SetOutputDirection(old_direction)
        self.upsample_image_filter.SetOutputSpacing(old_spacing)
        self.upsample_image_filter.SetSize(old_size)
        self.upsample_image_filter.SetInterpolator(self.order_upsampling)
        
        ds_image = self.downsample_image_filter.Execute(image)
        image = self.upsample_image_filter.Execute(ds_image)
        
        return image
        

class ImageContrastShiftTransform:
    def __init__(self, multiplier):
        self.multiplier = multiplier
        self.multiply_filter = ImageMultiplicativeBrightnessTransform(multiplier)

    def __call__(self, image):
        # SwigPyObjects, cannot be pickled for multiprocessing
        # Hence we define them in the call, not the initialization
        self.stats_image_filter = sitk.StatisticsImageFilter()
        self.clamp_image_filter = sitk.ClampImageFilter()
        
        self.stats_image_filter.Execute(image)
        mean_value = self.stats_image_filter.GetMean()
        mean_value = self.stats_image_filter.GetMean()
        max_value = self.stats_image_filter.GetMaximum()
        min_value = self.stats_image_filter.GetMinimum()
        image = image - mean_value
        image = self.multiply_filter(image)
        image = image + mean_value
        self.clamp_image_filter.SetLowerBound(min_value)
        self.clamp_image_filter.SetUpperBound(max_value)
        image = self.clamp_image_filter.Execute(image)
        return image




# TODO PETNoise