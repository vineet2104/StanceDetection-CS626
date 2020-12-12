# Project Title : Fake News Identification using Stance Detection

# Group Members:

Maheer Maloth (Roll Number - 180050054 )

Vineet Bhat ( Roll Number - 180260042 )

T Sanjev Vishnu ( Roll Number - 180110090 )

# Project Description:

With the advent of social media platforms, it has become extremely easy to propagate any information to the masses, thus making Fake News a potential threat to journalism and public discourse. The root cause of this problem lies in the fact that none of the social networking sites use any automatic system that can identify the veracity of news going around. This complex problem is broken into various stages and stance detection could prove to be a helpful building block.  

The idea of stance detection is simple. Given a news article, we determine the relevance of the body and the claim and classify the article accordingly into four categories: ‘agree’, ‘disagree’, ‘discuss’ and ‘unrelated’. Classes ‘disagree’ and ‘unrelated’ provide strong evidence that the news is fake, ‘agree’ indicates that the news is genuine.
A good stance detection solution would allow a human fact checker to retrieve the top articles that agree, disagree or discuss the claim in question, thus making the process fast and effective. It should also be possible to build a post-facto “truth-labelling’ system based on a stance detection system which would label a story as true or false based on stances of various news organizations on the topic. 

Through our project, we present a novel idea that combines the neural, statistical and external features to provide an efficient solution to this problem. The model measures the contextual and semantic similarity between the body and the headline.

# Steps to execute the code:

Download our repository on your system. Download the drive folder - https://drive.google.com/drive/folders/1ONkaX-abnk_4kvN1O0-ET7WwfoGRyZM2?usp=sharing which contains all the data files needed. Please rename this folder as 'Data' and save it in the same location where you have downloaded this repository.

For English:

1. To generate the necessary feature vectors, please run the notebook 'EnglishFeatureGeneratorFinal.ipynb'. The necessary packages needed have been mentioned in the notebook. 

2. To run our English model, please run the notebook 'English_Model_Final.ipynb'

For Bengali: 

1. To generate the necessary feature vectors, please run the notebook 'BengaliFeatureGeneratorFinal.ipynb'. The necessary packages needed have been mentioned in the notebook. 

2. To run our Bengali model, please run the notebook 'Bengali_Model_Final.ipynb'

For English Sentences Demo:

1. Please run the 'English_Demo.ipynb' notebook. The necessary packages needed have been mentioned in the notebook.

