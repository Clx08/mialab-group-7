import numpy as np
import SimpleITK as sitk
from scipy import ndimage

def binary_opening_3d(mask):
    #3d opening using a 3x3x3 cubic element to structure
    if mask.ndim != 3:
        raise ValueError("Mask must be a 3D array.")
    if mask.dtype != np.bool_:
        raise TypeError("Mask must be boolean type.")

    structure=np.ones((3,3,3), dtype=np.bool_)
    return ndimage.binary_opening(mask, structure=structure)

def binary_closing_3d(mask):
    #3d closing using a 3x3x3 cubic element to structure
    if mask.ndim != 3:
        raise ValueError("Mask must be a 3D array.")
    if mask.dtype != np.bool_:
        raise TypeError("Mask must be boolean type.")

    structure=np.ones((3,3,3), dtype=np.bool_)
    return ndimage.binary_closing(mask, structure=structure)

def opening_for_label(segmentation,label,background_label):
    #applies opening to just one label and reassign removed voxels to bckground
    if label==background_label:
        raise ValueError("Label must be different from background label")
    mask=segmentation==label
    opened_mask=binary_opening_3d(mask)

    result=segmentation.copy()
    absent_after_opening=np.logical_not(opened_mask)
    removed_voxels=np.logical_and(mask,absent_after_opening)
    result[removed_voxels]=background_label

    return result

def closing_for_label(segmentation,label,background_label):
    #applies closing to one label filling only background voxels
    if label==background_label:
        raise ValueError("Label must be different from background label")
    mask=segmentation==label
    closed_mask=binary_closing_3d(mask)

    available_background=segmentation==background_label
    added_voxels=np.logical_and(closed_mask,available_background)
    result=segmentation.copy()
    result[added_voxels]=label

    return result

def morphology_image(image,label,background_label,operation):
        #applies morphology to one label of 3d simpleitk segm
    if image.GetDimension()!=3:
        raise ValueError("Image must be 3D")
    if image.GetNumberOfComponentsPerPixel()!=1:
        raise ValueError("Image must be single channel")

    segmentation=sitk.GetArrayFromImage(image)
    if operation=="opening":
        result=opening_for_label(segmentation,label,background_label)
    elif operation=="closing":
        result=closing_for_label(segmentation,label,background_label)
    else:
        raise ValueError("Operation must be 'opening' or 'closing'")
    output_image=sitk.GetImageFromArray(result)
    output_image.CopyInformation(image)
    return output_image

if __name__ == "__main__":
    mask=np.zeros((15,15,15),dtype=bool)
    mask[4:11,4:11,4:11]=True
    mask[1,1,1]=True

    original=mask.copy()
    result=binary_opening_3d(mask)

    expected=original.copy()
    expected[1,1,1]=False

    assert result.shape==mask.shape, "Shape changed"
    assert result.dtype==np.bool_, "Output is not boolean"
    assert np.array_equal(result, expected), "Result doesn't match expected output"
    assert np.array_equal(mask,original), "Input was modified"

    print("Opening test passed")

    mask=np.zeros((15,15,15),dtype=bool)
    mask[4:11,4:11,4:11]=True
    expected=mask.copy()

    mask[7,7,7]=False
    original=mask.copy()

    result=binary_closing_3d(mask)

    assert result.shape==mask.shape, "Shape changed"
    assert result.dtype==np.bool_, "Output is not boolean"
    assert np.array_equal(result, expected), "Result doesn't match expected output"
    assert np.array_equal(mask,original), "Input was modified"

    print("Closing test passed")

    #synthetic segm:0=background, 1=other tissue, 3=selected tissue
    segmentation=np.zeros((15,15,15),dtype=np.uint8)
    segmentation[4:11,4:11,4:11]=3
    segmentation[1,1,1]=3
    segmentation[4:11,4:11,11:13]=1

    original=segmentation.copy()

    expected=segmentation.copy()
    expected[1,1,1]=0

    result=opening_for_label(segmentation,3,0)
    assert result.shape==segmentation.shape, "Shape changed"
    assert result.dtype==segmentation.dtype, "Label data type changed"
    assert np.array_equal(result, expected), "Result doesn't match expected output"
    assert np.array_equal(segmentation,original), "Input was modified"

    print("Opening for onelabel test passed")

    segmentation=np.zeros((15,15,15),dtype=np.uint8)
    segmentation[4:11,4:11,4:11]=3  

    #create bckgrnd hole and voxel belonging to other tissue
    segmentation[7,7,7]=0
    segmentation[7,7,8]=1

    original=segmentation.copy()
    expected=segmentation.copy()
    expected[7,7,7]=3

    result=closing_for_label(segmentation,3,0)
    assert result.shape==segmentation.shape, "Shape changed"
    assert result.dtype==segmentation.dtype, "Label data type changed"
    assert np.array_equal(result, expected), "Result doesn't match expected output"
    assert np.array_equal(segmentation,original), "Input was modified"

    print("Closing for onelabel test passed")


