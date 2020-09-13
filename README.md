Read me for MSc Project - William Richards

NOTE: THE CODE WILL NOT RUN AS IT REQUIRES a pretained affect net model and the processing and storage of many gigabytes of data. However, model results are printed in the jupyter notebooks.


The code is broken up in to 3 directories:

-Amigosnet: the code used to train the neural network

-Amigosdownload: the code used to download and process the raw amigos dataset to produce the dataset. The code for facial extraction and affect net pre training was removed to comply with university code rules.

- Jupyter notebooks: the Jupyter notebooks used to analyse final results




AMIGOSNET files:

train_model: File used to train a lost validation neural network using the optional parameters defined at the top. 

train_model_inter_subject: Used to train an inter subject validation model and save results using optional parameters 

Amigosdataset: contains lazy pytorch datasets and data loaders for the inter-subject and lost models as well as a method to run one epoch for any network and a function to evaluate the performance of a model.

Resize: Contains code used to resize the extracted facial frames to a standard resolution.
 The input and output directories are defines in the file.

Models/Vggface: contains definitions of the VGG and adapted VGG architecture written using the pytorch library as well as methods to initialise the models with the desired weights. 


Amigosdownload files:

download_data: contains a python class to download all short individualdb videos, convert them to raw frames and delete original videos.

download_large: contains a python class to download all long individualdb videos, convert them to raw frames and delete original videos.



Notebooks files:

Average annotations: Used to average annotator annotations and create a csv containing target labels which are used in the Amigosdataset file.

all_analysis: notebook used to calculate model scores for each loso/intersubject run. Example Plot of histogram included.

split_analysis: example of histogram and plot of model emotion reproduction over time



