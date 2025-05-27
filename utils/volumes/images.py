import SimpleITK as sitk


def load_image(image_path, dtype=None):
    if dtype is not None:
        return sitk.ReadImage(image_path, outputPixelType=dtype)
    else:
        return sitk.ReadImage(image_path)


def apply_numpy_fn(image, fn, output_is_image=True):
    image_np = sitk.GetArrayFromImage(image)
    out = fn(image_np)
    if output_is_image:
        image_out = sitk.GetImageFromArray(out)
        image_out.CopyInformation(image)
        return image_out
    else:
        return out


def resample_image(image, to):
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetOutputDirection(to.GetDirection())
    resampler.SetOutputOrigin(to.GetOrigin())
    resampler.SetSize(to.GetSize())
    resampler.SetOutputSpacing(to.GetSpacing())
    resampled_image = resampler.Execute(image)
    return resampled_image
    


if __name__ == "__main__":
    pass # TODO: make test images (CT, PT, MRI)