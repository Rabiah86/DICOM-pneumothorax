import numpy as np #adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import pandas as pd #data structures and operations for manipulating numerical tables and time series
import sys # system path io
import pydicom #manage DICOM files, DICOM format can store these images with metadata (patient id, age, sex, etc.) in one file with .dcm extension.
from glob import glob #glob module is used to retrieve files/pathnames matching a specified pattern
from tqdm import tqdm # show progress bar when a loop is running
from mask_functions import rle2mask, mask2rle # Kaggle functions for manipulating RLEs (Run-length encoding (RLE) is a very simple form of lossless data compression)

sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation') #set the directory path
rles_df = pd.read_csv('../input/siim-train-test/siim/train-rle.csv')
# Set the column names manually, just in case, because sometimes leading spaces or typos could cause errors!
rles_df.columns = ['ImageId', 'EncodedPixels']

#turn your dataset to a dictionary, to define the columns and combine X-ray information with corresponding metadata
def dicom_to_dict(dicom_data, file_path, rles_df, encoded_pixels=True):
    """Parse DICOM dataset and returns a dictonary with relevant fields.

    Args:
        dicom_data (dicom): chest x-ray data in dicom format.
        file_path (str): file path of the dicom data.
        rles_df (pandas.core.frame.DataFrame): Pandas dataframe of the RLE.
        encoded_pixels (bool): if True we will search for annotation. So true for train data, flase for test!
        
    Returns:
        dict: contains metadata of relevant fields.
    """
    
    data = {}
    
    # Parse fields with meaningful information
    data['patient_name'] = dicom_data.PatientName
    data['patient_id'] = dicom_data.PatientID
    data['patient_age'] = int(dicom_data.PatientAge)
    data['patient_sex'] = dicom_data.PatientSex
    data['pixel_spacing'] = dicom_data.PixelSpacing
    data['file_path'] = file_path
    data['id'] = dicom_data.SOPInstanceUID
    
    # look for annotation if enabled (train set)
    if encoded_pixels:
        encoded_pixels_list = rles_df[rles_df['ImageId']==dicom_data.SOPInstanceUID]['EncodedPixels'].values
       
        pneumothorax = False
        for encoded_pixels in encoded_pixels_list:
            if encoded_pixels != ' -1':
                pneumothorax = True
        
        # get meaningful information (for train set)
        data['encoded_pixels_list'] = encoded_pixels_list
        data['has_pneumothorax'] = pneumothorax
        data['encoded_pixels_count'] = len(encoded_pixels_list)
        
    return data
# create a list of all the files
train_fns = sorted(glob('../input/siim-train-test/siim/dicom-images-train/*/*/*.dcm'))
# parse train DICOM dataset
train_metadata_df = pd.DataFrame()
train_metadata_list = []
for file_path in tqdm(train_fns):
    dicom_data = pydicom.dcmread(file_path)
    train_metadata = dicom_to_dict(dicom_data, file_path, rles_df)
    train_metadata_list.append(train_metadata)
train_metadata_df = pd.DataFrame(train_metadata_list)

