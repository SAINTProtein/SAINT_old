# SAINT
This repository contains the official implementation of our paper **SAINT**: **S**elf-**A**ttention Augmented **I**nception-Inside-Inception **N**e**t**work Improves Protein Secondary Structure Prediction.

# Download the dataset
##### CB6133 dataset and CB513 dataset
Please download the files *cullpdb+profile_5926_filtered.npy.gz* and *cb513+profile_split1.npy.gz* from [this website](http://www.princeton.edu/~jzthree/datasets/ICML2014/) for training the model on the filtered version of CB6133 dataset(duplicates removed) and testing it on CB513 dataset, then rename the files as **psp.npy.gz** and **CB513.npy.gz** respectively:
##### CASP10 and CASP11 dataset
To test the model on CASP10 and CASP11 benchmark-datasets, please download the files from [this website](https://drive.google.com/drive/folders/1404cRlQmMuYWPWp5KwDtA7BPMpl-vF-d).

Finally, put all the downloaded files in the *./SAINT* folder.

# Setup:
The implementation is in python3. Keras 2.2.4 and Tensorflow 1.13.1 were used to build this project. To intall these dependencies please run the following commands(You can use [Anaconda](https://www.anaconda.com/) or [Pip](https://pip.pypa.io/en/stable/installing/) for installation):
##### For pip
> pip install tensorflow-gpu

> pip install keras

##### For Anaconda
> conda install tensorflow-gpu

> conda install keras

# Training the model
To train the model after cloing the repository please run the following commands:
> cd SAINT

> python3 train.py

Various parameters can be changed for training or evaluating the model in the file *./SAINT/config.py*

# Running pretrained model
##### Evaluation
To run the pretrained model please download the file *saint_pssp.h5* from [this link](https://drive.google.com/open?id=1dV5T1VUzVzU8qJD1W6wEEET28eTIPKfM) and place it in the *./SAINT* folder.
Then run the following command:
> python3 evaluate.py
##### Attention weights visualization
In order to visualize the attention distribution please run the following command while inside the *./SAINT* folder:
> python3 visualize_attention_distribution.py
