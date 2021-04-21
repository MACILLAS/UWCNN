# An Incremental improvement to underwater scene prior inspired deep underwater image and video enhancement: UWCNN++
### By: Max Midwinter
### Date: 2021-04-21

This work is based on: 

Underwater scene prior inspired deep underwater image and video enhancement
DOI: https://doi.org/10.10.16/j.patcog.2019.107038

Please see the attached PDF for details about UWCNN++

#### Using the Code
In the UWCNN++ directory is a sample of how the project is structured.
* NYU_GT is a stand in for the NYUv2 RGBD dataset
* NYU_UW_GT is the output for cropped NYU_GT images by using resizeNYUGT in preprocess.py
* NYU_UW_type1 is the output folder for generated synthetic images (you'll need to create this folder prior to running generate_image_underwater_v3.py)
* results is the directory that UWCNN will save the images you run with model_test
* save_model is where the trained CNN model will be saved
* test_images is a directory where we store real underwater images we want to test
* train_type1 is the directory where train_model will save checkpoints (please also create this directory prior to training)

UWCNN++ also contains 4 main scripts that will generally used in the following order
* generate_image_underwater_v3.py that generates the underwater images (lines: 33, 48) need to be changed manually to generate different turbidity levels
* preprocess.py the main function of this function is to generate the csv files that UWCNN will use to create the database object for training
* UWCNN.py is the network and contains the training and testing code 
* hsi_normalize.py is used to enrich the colour of the network, this will read all the images in results directory and will overwrite them

#### Results
See pdf