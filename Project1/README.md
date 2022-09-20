This folder contains all the code used for the first project of the special curriculum, which is based on the Higgs Machine Learning Challenge from 2014.
The test and training data sets are too large to compress and upload to github, but can be downloaded from https://www.kaggle.com/competitions/higgs-boson/data .
The AMS.py file will use the preprocessed training file to generate test labels that fits the training data. 

The preproc.py file will be needed in order to turn this data into the correct format and normalize it for use in the other codes. The pathing in this file is
made for the user who uploaded it, so one would have to change it to an appropriate path in order to use it. 

The code for the different model builds can be found in model1.py, model2.py and model3.py. The validation.py file uses our first model along with the validation
data sets generated in the preproc.py file. 

The Higgs_test.py file tests our final model against the test data set. The logs of all the runs can be found in the log folder, which contains the accuracy and loss
for training and validation/training set at each epoch. The plots for these files are found in the plots folder along with the python code used to generate them.

The model we end up with is saved in the my_model folder.
