# StanceDetection-CS626
Course project for CS-626 at IIT Bombay

This work is based on the paper "On the Benefit of Combining Neural, Statistical and External Features for Fake News Identification" - Bhatt et al, 2017. You can view the paper here - https://arxiv.org/pdf/1712.03935.pdf 

Dataset Link: Download the repository - https://github.com/FakeNewsChallenge/fnc-1

Due to memory issues and limitations of the pre-trained skip thought vector generator tool, the train set has been limited to 26064 headline-body pairs and test set has been limited to 13927 headline-body pairs


## Current Status(please add the work that you have done till now under this section):

1. Dataset creation and exploration notebooks added: Please run CreateDataset.ipynb first to create a complete train and test set. Then run the DataAnalysis.ipynb to visualize the data. Once these two steps have been completely, you would have two '.csv' files containing all the required train and test data. You can find these two '.csv' files in this folder - https://drive.google.com/drive/folders/1vXRzjXrCXYeChqMn43jS_AgzRVLfBAKK?usp=sharing

2. Added NeuralFeatureGenerator.ipynb and StatisticalFeatureGenerator.ipynb. NeuralFeatureGenerator.ipynb - creates a neural embedding vector using Skip-thoughts. For each headline-body pair, a feature vector of dimension (1,4800) is generated. StatisticalFeatureGenerator.ipynb - creates a statistical vector based on Term frequencies. For each headline-body pair, a feature vector of dimension (1,10000) is generated. You would have to run the NeuralFeatureGenerator.ipynb notebook in a different environment because of the dependencies. But for the time being, you dont have to run any generator code since all the corresponding .npy files have been uploaded in the google drive folder mentioned in point 1

