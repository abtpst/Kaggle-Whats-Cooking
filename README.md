# Kaggle-Whats-Cooking

Here we are going to solve the What's Cooking challenge on Kaggle

https://www.kaggle.com/c/whats-cooking

I will be working with python's `sci-kit learn` library.
I am going to use `TfidfVectorizer` for feature selection and `LogisticRegression` classifier for classification. I am going to combine these two components in python's `GridSearchCV ` `Pipeline`

Python's `GridSearchCV` is a very useful package for building and training machine learning models. We will use this packge to

> - Build a pipeline consisting of a feature selector and a classifier.
> - Provide a set of options/parameters for the pipeline's constituents.
> - Store the best set of parameters that GridSearchCV comes up with.
> - Use the optimal parameters in the above step to build an optimal pipeline for predictions.


### Structure

1. **data** 
    This folder has the training and test data.
2. **picks**
    This folder is used to store intermediate results such as the best set of parameters determined by  GridSearchCV. 
3. **results**
    This folder is used to store the feedback loop results and the final predictions.
4. **src**
    This folder contains all of the source code.
    
### Training

Please look at the well documented `cookTrain.py` script in the **src** folder.
Here is the flow of events
  

### Predict

Please look at the well documented cookPredict.py

