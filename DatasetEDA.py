import numpy as np #adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import pandas as pd #data structures and operations for manipulating numerical tables and time series
import sys # system path io
import pydicom #manage DICOM files, DICOM format can store these images with metadata (patient id, age, sex, etc.) in one file with .dcm extension.
from glob import glob #glob module is used to retrieve files/pathnames matching a specified pattern
from tqdm import tqdm # show progress bar when a loop is running
from mask_functions import rle2mask, mask2rle # Kaggle functions for manipulating RLEs (Run-length encoding (RLE) is a very simple form of lossless data compression)
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly import tools
from plotly.graph_objs import *
from plotly.graph_objs.layout import Margin, YAxis, XAxis
init_notebook_mode()

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
    
    data = {} # data is a dictionary
    
    # Parse fields with meaningful information
    data['patient_name'] = dicom_data.PatientName
    data['patient_id'] = dicom_data.PatientID
    data['patient_age'] = int(dicom_data.PatientAge)
    data['patient_sex'] = dicom_data.PatientSex
    data['pixel_spacing'] = dicom_data.PixelSpacing
    data['file_path'] = file_path # the path is saved, cz it will be waste of memory if we save the file!
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
train_metadata_df = pd.DataFrame(train_metadata_list)  # at this point I have all the training insances and metadata stored here!
# create a list of all the files
test_fns = sorted(glob('../input/siim-train-test/siim/dicom-images-test/*/*/*.dcm'))
# parse test DICOM dataset
test_metadata_df = pd.DataFrame()
test_metadata_list = []
for file_path in tqdm(test_fns):
    dicom_data = pydicom.dcmread(file_path)
    test_metadata = dicom_to_dict(dicom_data, file_path, rles_df, encoded_pixels=False)
    test_metadata_list.append(test_metadata)
test_metadata_df = pd.DataFrame(test_metadata_list) # at this point I have all the test insances and metadata stored here!
###########################################EDA begins here!#########################################################
###########1- Explore the data, for some datatypes, the file contents are not visible like DICOM, so always select a subset look at it and try to understand more
import matplotlib.pyplot as plt
from matplotlib import patches as patches
num_img = 4
subplot_count = 0
fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10)) #cz we're showing 4 subplots in the figure!   Share the Y axis! the X axis is a proportion of the number of images!
for index, row in train_metadata_df.sample(n=num_img).iterrows(): # get a random sample of 4 images
    dataset = pydicom.dcmread(row['file_path']) 
    ax[subplot_count].imshow(dataset.pixel_array, cmap=plt.cm.bone) # set the color map to bone! show the pixel array of the dicom image
    # label the x-ray with information about the patient
    ax[subplot_count].text(0,0,'Age:{}, Sex: {}, Pneumothorax: {}'.format(row['patient_age'],row['patient_sex'],row['has_pneumothorax']),
                           size=26,color='white', backgroundcolor='black')
    subplot_count += 1
##########2- show the ROIs#########################################################################################
"""
When the ROIs are visible to the developer, 
1- then its easier to check if the ROIs are visible, if not then we can take better decisions on what filters we need to apply like the case with this dataset, CLAHE made the image look much better 
so this helps you make informed decisions

2- can spot some insights, like confusion caused by the patient gender, like in chest Xrays
3- also can detect any BB on the top of each other
"""
def bounding_box(img):
    # return max and min of a mask to draw bounding box
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]] # [0,-1] means from the first index to the last index.. cz rememner here indexing begins from 0 
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def plot_with_mask_and_bbox(file_path, mask_encoded_list, figsize=(20,10)):
    
    import cv2
    
    """Plot Chest Xray image with mask(annotation or label) and without mask.

    Args:
        file_path (str): file path of the dicom data.
        mask_encoded (numpy.ndarray): Pandas dataframe of the RLE.
        
    Returns:
        plots the image with and without mask.
    """
    
    pixel_array = pydicom.dcmread(file_path).pixel_array
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    clahe_pixel_array = clahe.apply(pixel_array)
    
    # use the masking function to decode RLE
    mask_decoded_list = [rle2mask(mask_encoded, 1024, 1024).T for mask_encoded in mask_encoded_list]
    
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(20,10))
    
    # print out the xray
    ax[0].imshow(pixel_array, cmap=plt.cm.bone)
    # print the bounding box
    for mask_decoded in mask_decoded_list:
        # print out the annotated area
        ax[0].imshow(mask_decoded, alpha=0.3, cmap="Reds")
        rmin, rmax, cmin, cmax = bounding_box(mask_decoded)
        bbox = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')
        ax[0].add_patch(bbox)
    ax[0].set_title('With Mask')
    
    # plot image with clahe processing with just bounding box and no mask
    ax[1].imshow(clahe_pixel_array, cmap=plt.cm.bone)
    for mask_decoded in mask_decoded_list:
        rmin, rmax, cmin, cmax = bounding_box(mask_decoded)
        bbox = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')
        ax[1].add_patch(bbox)
    ax[1].set_title('Without Mask - Clahe')
    
    # plot plain xray with just bounding box and no mask
    ax[2].imshow(pixel_array, cmap=plt.cm.bone)#draws an image on the current figure 
    for mask_decoded in mask_decoded_list:
        rmin, rmax, cmin, cmax = bounding_box(mask_decoded)
        bbox = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')
        ax[2].add_patch(bbox)
    ax[2].set_title('Without Mask')
    plt.show() #plt.show() displays the figure 
    #Calling plt.show() before you've drawn anything using imshow doesn't make any sense.
    # lets take 10 random samples of x-rays with 
train_metadata_sample = train_metadata_df[train_metadata_df['has_pneumothorax']==1].sample(n=10)
print(len(train_metadata_sample))
# plot ten xrays with and without mask
for index, row in train_metadata_sample.iterrows():
    file_path = row['file_path']
    mask_encoded_list = row['encoded_pixels_list']
    print('image id: ' + row['id'])
    plot_with_mask_and_bbox(file_path, mask_encoded_list)
########################3- plot statistics and histograms #######################
#find any imbalance
#see if more than one annotation per instance 
#see if any missing annotations
    
    # print missing annotation
missing_vals = train_metadata_df[train_metadata_df['encoded_pixels_count']==0]['encoded_pixels_count'].count()
print("Number of x-rays with missing labels: {}".format(missing_vals))

nok_count = train_metadata_df['has_pneumothorax'].sum() # how many have the pnomx
ok_count = len(train_metadata_df) - nok_count #how many are healthy
x = ['No Pneumothorax','Pneumothorax'] # name the rows
y = [ok_count, nok_count] # name the variables for the columns
trace0 = Bar(x=x, y=y, name = 'Ok vs Not OK') #legend and figure that plots the totals
nok_encoded_pixels_count = train_metadata_df[train_metadata_df['has_pneumothorax']==1]['encoded_pixels_count'].values #how many infected areas are there in each instance?
trace1 = Histogram(x=nok_encoded_pixels_count, name='# of annotations') #value against count, so its a histogram!
fig = tools.make_subplots(rows=1, cols=2)
fig.append_trace(trace0, 1, 1)#trace,row,column
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=400, width=900, title='Pneumothorax Instances')
iplot(fig)
##############4- Explore Age, Sex, and Pneumothorax#######################
"""
1- Remember the lung grows as we grow, so younger means smaller ROIs
2- can have a better idea about the sizes from the aages 
3- can also check for anomolies, do all values make since? if some ppl seem to be like 400 years old just remove such anomolies before you continue
4- The median age for people who has pneumothorax is not considerably but is slightly lower.
more observations:
We have slightly more instanes of male x-rays.
Male and female are equally likely to have Pneumothorax (at least according to our dataset)
"""
pneumo_pat_age = train_metadata_df[train_metadata_df['has_pneumothorax']==1]['patient_age'].values
no_pneumo_pat_age = train_metadata_df[train_metadata_df['has_pneumothorax']==0]['patient_age'].values
pneumothorax = Histogram(x=pneumo_pat_age, name='has pneumothorax')
no_pneumothorax = Histogram(x=no_pneumo_pat_age, name='no pneumothorax')
fig = tools.make_subplots(rows=1, cols=2)
fig.append_trace(pneumothorax, 1, 1)
fig.append_trace(no_pneumothorax, 1, 2)
fig['layout'].update(height=400, width=900, title='Patient Age Histogram')
iplot(fig)

### Box chart for age 
trace1 = Box(x=pneumo_pat_age, name='has pneumothorax')
trace2 = Box(x=no_pneumo_pat_age[no_pneumo_pat_age <= 120], name='no pneumothorax') # to remove anomolies
data = [trace1, trace2]
iplot(data)
###########GENDER###
train_male_df = train_metadata_df[train_metadata_df['patient_sex']=='M']
train_female_df = train_metadata_df[train_metadata_df['patient_sex']=='F']
male_ok_count = len(train_male_df[train_male_df['has_pneumothorax']==0])
female_ok_count = len(train_female_df[train_female_df['has_pneumothorax']==0])
male_nok_count = len(train_male_df[train_male_df['has_pneumothorax']==1])
female_nok_count = len(train_female_df[train_female_df['has_pneumothorax']==1])
ok = Bar(x=['male', 'female'], y=[male_ok_count, female_ok_count], name='no pneumothorax')
nok = Bar(x=['male', 'female'], y=[male_nok_count, female_nok_count], name='has pneumothorax')

data = [ok, nok]
layout = Layout(barmode='stack', height=400)

fig = Figure(data=data, layout=layout)
iplot(fig, filename='stacked-bar')
#####noe donut chart for the percentages
m_pneumo_labels = ['no pneumothorax','has pneumothorax']
f_pneumo_labels = ['no pneumothorax','has pneumothorax']
m_pneumo_values = [male_ok_count, male_nok_count]
f_pneumo_values = [female_ok_count, female_nok_count]
colors = ['#FEBFB3', '#E1396C']
# original source code: https://plot.ly/python/pie-charts/#donut-chart

fig = {
  "data": [
    {
      "values": m_pneumo_values,
      "labels": m_pneumo_labels,
      "domain": {"column": 0},
      "name": "Male",
      "hoverinfo":"label+percent",
      "hole": .4,
      "type": "pie"
    },
    {
      "values": f_pneumo_values,
      "labels": f_pneumo_labels,
      "textposition":"inside",
      "domain": {"column": 1},
      "name": "Female",
      "hoverinfo":"label+percent",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Pneumothorax - Male vs Female",
        "grid": {"rows": 1, "columns": 2},
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Male",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Female",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
iplot(fig)



