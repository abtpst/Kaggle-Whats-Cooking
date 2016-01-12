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

Please look at the well documented `cookTrain.py` script in the **cook** package inside **src**.
Here is the flow of events

1. Create   `GridSearchCV` `Pipeline` comprising of `TfidfVectorizer` and `LogisticRegression` classifier
2.  Load training data, perform cleanup on the relevant columns. For our purpose, this will be the `ingredients` column. Finally, create training and validation sets
3.  Fit the training set on the pipeline
4.  Calculate the best set of parameters and make predictions on the validation set
5.  Evaluate metrics and scores for the pipeline's `best_estimator`
6.  Document the prediction results on the validation set and create a `pandas` `DataFrame` for feedback 
7.  Extract the 'mistakes' from the validation set predictions and re-train the pipeline. This is our feedback loop.
8.  Once again, make predictions on the validation set and evaluate metrics and scores for the pipeline's `best_estimator`
9.  Store the best set of parameters

### Predict
Please look at the well documented `cookPredict.py` script in the **cook** package inside **src**.
Here is the flow of events

1. Create   `GridSearchCV` `Pipeline` comprising of `TfidfVectorizer` and `LogisticRegression` classifier. This time we use the best set set of parameters obtained from training.
2.  Load test data and perform cleanup on the relevant columns. For our purpose, this will be the `ingredients` column. 
3.  Make predictions on the test data and store results

### What I Observed

Here are the stats that i observed before and after feedback


    Fitting 3 folds for each of 300 candidates, totalling 900 fits
    [Parallel(n_jobs=3)]: Done  44 tasks      | elapsed:  4.6min
    [Parallel(n_jobs=3)]: Done 194 tasks      | elapsed: 23.1min
    [Parallel(n_jobs=3)]: Done 444 tasks      | elapsed: 71.4min
    [Parallel(n_jobs=3)]: Done 794 tasks      | elapsed: 164.7min
    [Parallel(n_jobs=3)]: Done 900 out of 900 | elapsed: 194.8min finished
    best score: 0.779
    best parameters set:
    	 clf__C: 10
    	 vect__max_df: 0.7
    	 vect__ngram_range: (1, 1)
    	 vect__use_idf: True
    ('Accuracy:', 0.78521746417497695)
    Fitting 3 folds for each of 300 candidates, totalling 900 fits
    [Parallel(n_jobs=3)]: Done  44 tasks      | elapsed:  3.9min
    [Parallel(n_jobs=3)]: Done 194 tasks      | elapsed: 17.7min
    [Parallel(n_jobs=3)]: Done 444 tasks      | elapsed: 54.2min
    [Parallel(n_jobs=3)]: Done 794 tasks      | elapsed: 131.1min
    [Parallel(n_jobs=3)]: Done 900 out of 900 | elapsed: 153.2min finished
    best score: 0.719
    best parameters set:
    	 clf__C: 20
    	 vect__max_df: 0.6
    	 vect__ngram_range: (1, 2)
    	 vect__use_idf: True
    ('Accuracy:', 0.73130892348169263)

1. It is evident that the best set of parameters has changed after feedback. Hence, the classifier did 'learn' something.
2.  Note that the accuracy actually decreases when we use the new set of parameters. This is probably due to `overfitting`.
3.  Nonetheless, when i submitted my results on Kaggle, i got about 78% accuracy.
