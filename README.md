# NF_detection_model
Perform the detection of skin-surface level neurofibromas with a deep learning model.

This script will contains the function segment_images(input_file_name) that takes the input input_file_name in the form of a string.


Inputs:

* input_file_name (str) - the path to the folder with input images


Outputs, located in output folder:

* "output/Input Images/" - input images in their resized form and with label corresponding to their outputs in other folders
* "output/Segmentation Masks/" - the mask of the model's prediction for lcoation of neurofibromas
* "output/Segmented Images/" - the segmentation mask overlayed on the corresponding input image


Install required packages that aren't pre-installed with Python:

`pip install tensorflow`

`pip install pillow`

`pip install opencv-python`
