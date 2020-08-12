"""Module for getting spectra from ICCD images

Made new iccd3.py from iccd2.py on 2016/06/19

"""

print("IMPORTING STANDARD PYTHON MODULES...")
import numpy as np
import scipy.ndimage as spimg
import h5py
from scipy import interp
from scipy.stats import scoreatpercentile
import scipy.signal
import os
print("DONE")

import LB51_calibman

#### CONSTANTS
ICCD_IMAGE_SIZE = 1024  # Number of rows in ICCD image, also number of columns in ICCD image
# Path to LCLS HDF5 files:
HDF5_PATH = '/reg/data/ana01/amo/amob5114/hdf5/amob5114-r'
# ICCD Images key in LCLS data:
ICCD_IMAGES_STR = 'Configure:0000/Run:0000/CalibCycle:0000/Pimax::FrameV1/AmoEndstation.0:Pimax.0/data'
# The default background ROI:
DEF_BG_ROI = np.ix_(np.append(np.arange(100, 300), np.arange(700, 900)), 
                    np.arange(500, 700))
#### END CONSTANTS

def get_run_set_spectra(run):
    cals = LB51_calibman.get_cals(run)
    bg_img = get_bg_image(cals['iccdBGRun'])
    data_list = []
    for run in cals['all_runs']:
        print('Processing Run '+str(run))
        data_list.append(get_spectra(run, bg_img, cals))
    data = np.hstack(data_list)
    return data

def get_spectra(run, bg_img, cals):
    """Return spectra retrieved from LCLS HDF5 file
    """
    # Initialization:
    no_sam_specs = []
    sam_specs = []
    iccd_images = get_iccd_images(run)
    for current_img in iccd_images:
        spec_image = do_bg_correction(current_img, bg_img, cals['burst_mode'])
        no_sam_spec, sam_spec = get_spectra_from_image(spec_image, cals['shear'], cals['no_sam_spec_ROI'], cals['sam_spec_ROI'])
        no_sam_specs.append(no_sam_spec)
        sam_specs.append(sam_spec)
    data_dict = {'sam_spec': np.array(sam_specs),
                 'no_sam_spec': np.array(no_sam_specs),
                 'run_num': np.array([run]*len(iccd_images)),
                 'image_num': np.arange(len(iccd_images))}
    dtype_args = [(key, data_dict[key].dtype.type, np.size(data_dict[key][0])) for key in data_dict.keys()]
    data_dtype = np.dtype(dtype_args)
    data = np.array(zip(*data_dict.values()), dtype=data_dtype)
    return data

def get_spectra_from_image(imgIn, shear, noSamROI, samROI) :
    """Shear imgIn, then extracts spectra from the ROIs"""
    # The simplest method would be to shear the entire image, then
    # extract the desired spectra, but we just shear only the region
    # corresponding to the spectra since this turns out to be faster
    noSamROIBig = np.copy(noSamROI)
    noSamROIBig[2] = 0
    noSamROIBig[3] = 1024
    noSamSpec = get_roi(imgIn, noSamROIBig)
    noSamSpec = shear_image(shear, noSamSpec, noSamROI[0])
    noSamSpec = noSamSpec[:, noSamROI[2]:noSamROI[3]]
    noSamSpec = np.sum(noSamSpec, 0)
    samROIBig = np.copy(samROI)
    samROIBig[2] = 0
    samROIBig[3] = 1024
    samSpec = get_roi(imgIn, samROIBig)
    samSpec = shear_image(shear, samSpec, samROI[0])
    samSpec = samSpec[:, samROI[2]:samROI[3]]
    samSpec = np.sum(samSpec, 0)
    return noSamSpec, samSpec

def get_bg_image(bg_run):
    """Return bg image as cosmic ray subtracted averaged image of a dark run
    """
    bg_run_images = get_iccd_images(bg_run)
    bg_run_images = [do_cosmic_sub(bg_img) for bg_img in bg_run_images]
    bg_image = np.average(bg_run_images, axis=0)
    return bg_image

def get_iccd_images(run):
    runStr = get_run_str(run)
    dataFile = h5py.File(runStr, 'r')
    iccdImages = np.array(list(dataFile[ICCD_IMAGES_STR]))
    dataFile.close()
    return iccdImages

def do_bg_correction(dataImg, bgImg, bgROI=DEF_BG_ROI, burst=True):
    """Corrects the background in an ICCD image
    """
    dataImg = dataImg-bgImg
    dataImg = do_cosmic_sub(dataImg)
    if burst:
        #Subtract typical background column
        y = np.arange(0, 1024)
        x = np.arange(750, 800)
        indicies = np.ix_(y, x)
        bgColROI = dataImg[indicies]
        bgCol = np.mean(bgColROI, 1)
        bgColInterped = interp(np.arange(ICCD_IMAGE_SIZE).reshape(1024, 1), y, bgCol)
        imgOut = dataImg - bgColInterped
    else :
        #subtract scaled background image
        dataBGROI = dataImg[bgROI]
        dataROIMed = np.median(dataBGROI)
        bgBGROI = bgImg[bgROI]
        bgROIMed = np.median(bgBGROI)
        weight = dataROIMed/bgROIMed
        imgOut = dataImg-bgImg*weight
    return imgOut

def do_cosmic_sub(img, bgROI=DEF_BG_ROI, thresholdMultiplier=5) :
    """Correct the cosmic rays/other random high-valued pixels in an ICCD image
    """
    imgROIData = img[bgROI]
    pix95 = scoreatpercentile(imgROIData.flatten(), 95)
    pix2 = scoreatpercentile(imgROIData.flatten(), 2)
    threshold = (pix95-pix2)*thresholdMultiplier
    imgMedFilt = scipy.signal.medfilt(img,3)
    # Boolean index of the bad pixels of the image:
    bad = np.abs(img-imgMedFilt) / threshold > 1.0       
    newImg = img.copy()
    newImg[bad] = imgMedFilt[bad]
    return newImg

def shear_image(shear, imgIn, firstRowIndex=0) :
    """Shears the image
        firstRowIndex -- If the passed image is a smaller slice of
            the total spectrometer image, then this should be the
            first row contained in the slice, indexed in the usual python
            way where the first row of the total image has index 0
    The input image is in coordinate system of the spectrometer as
        (axis 0 index = vertical row #, axis 1 index = horizontal column #).
        The sheared output image is in the coordinate system of
        (axis 0 index = vertical row #,
        axis 1 index = horizontal column # + (row #+firstRowIndex-ICCD_IMAGE_SIZE)*shear,
        such that the ICCD_IMAGE_SIZE'th column is left unsheared
    """
    def shear_func(output_coords):
        """Returns the coordinates of the sheared image""" 
        input_coords = (output_coords[0],
                        output_coords[1]+(output_coords[0]+firstRowIndex-ICCD_IMAGE_SIZE)*shear)
        return input_coords
    imgOut = spimg.geometric_transform(imgIn, shear_func, order=1)
    return imgOut

def get_run_str(run) :
    """Return the hdf5 path of run number run"""
    runStr = HDF5_PATH + str(int(run)).zfill(4) + '.h5'
    return runStr

def get_roi(img, roiEdges) :
    """Return roiData = img[roiEdges[0]:roiEdges[1], roiEdges[2]:roiEdges[3]]
    """
    roiData = img[roiEdges[0]:roiEdges[1], roiEdges[2]:roiEdges[3]]
    return roiData
