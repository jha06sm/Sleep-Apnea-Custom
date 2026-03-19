# Sleep-Apnea

# Preparations
## Dataset
The data is available at [Apnea-ECG Database](https://physionet.org/content/apnea-ecg/1.0.0/). Please download and extract the file to `dataset`.
## Downloading dependencies
```
Python 3.10.12
Keras 2.12.0
TensorFlow 2.12.0
```
# Preprocessing
Please run the file `preprocessing.py` to extract the data for CNN's input. In our method, we also design ablation study with show as below: 
![window_size](https://github.com/vinuni-vishc/MPCNN-Sleep-Apnea/assets/104493696/02acff41-19a6-4859-bc3d-254694af24fb)

The design of the ablation studys with different window size. 

$T_1$: From P peak to the end of ST segment. 

$T2$: From P peak to the end of QRS segment. 

$T3$: Q peaks to the end of ST segment. 

$T4$: QRS segment.

Read the comment in `preprocessing.py` to get the file data for each situation. $T_1$ is the case that achieved the highest performance. The extracted file has the '.pkl' format.

# Deep Learning evaluation

## Per-recording classification
After preprocessing, we will test our method using three different deep learning architectures: modified LeNet-5, BAFNet, and SEMSCNN. Please refer to the paper for more information about each model. The performance was evaluated on Google Colab, and the files are in the ".ipynb" format. Change the filename from ".pkl" to test the results for each dataset (from $T_1$ to $T_4$). There are three files including: `BAFNET_model.ipynb`, `LeNet5_model.ipynb`, and `SE-MSCNN_model.ipynb`. Additionally, read the comments in each file to understand how to modify the code to conduct other experiments to test the effectiveness of each term in the feature extraction process of MPCNN: MinDP, MaxDP, and MeanDP (from $M_1$ to $M_7$).

![Screenshot 2023-11-12 162043](https://github.com/vinuni-vishc/MPCNN-Sleep-Apnea/assets/104493696/76de8b69-31a8-4306-bc28-aa51d2e22a1f)

## Per-segment classification
After finishing the per-recording classification and extracting the CSV file, go to `test_per_recording.py` and enter the name of the '.csv' file as instructed in the comments to obtain the results of the per-segment classification.



.
