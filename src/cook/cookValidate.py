'''
Created on Dec 31, 2015

@author: abtpst
'''
import pandas as pd
import cookUtil as cookut
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.cross_validation import train_test_split
'''
In this method I incorporates the feedback data with the original training set.
Then I create training and validation sets. Next I fit the sklearn pipeline
with this new training set. Finally, I calculate the best of parameters and 
make predictions on the validation set

Arguments:
pipeline <=> sklearn Pipeline with TfidfVectorizer and LogisticRegression
parameters <=> Parameters for initializing Pipeline components
ultimateTraindf <=> DataFrame having ingredients and cuisine columns from the training set

Returns:
bestParameters <=> Best set of parameters for GridSearchCv Pipeline after retraining with feedback data
predictions <=> Predictions on the validation set
'''
def feedback(pipeline,parameters,ultimateTraindf):
	# Read feedback data
    fbkdf = pd.read_csv("../../results/feedback.csv")
    # Extract incorrect predictions
    fbkdf = fbkdf.loc[fbkdf['check'] == False]
	# Combine with rest of the training data
    fbkdf = fbkdf.append(ultimateTraindf, ignore_index=True)
    # Create matrix for learning and prediction
    X, y = fbkdf['properIngredients'], fbkdf['cuisine'].as_matrix()
    # Split further into training and validation sets
    Xtrain, Xvalidate, ytrain, yValidate = train_test_split(X, y, train_size=0.7)
    # Initialize gridSearchClassifierCV Classifier with parameters
    gridSearchClassifier = cookut.getGridSearchCv(pipeline, parameters)
    # Fit the gridSearchClassifier on Training Set
    gridSearchClassifier.fit(Xtrain,ytrain)
    # Calculate best set of parameters and make predictions on validation set
    return validate(parameters, gridSearchClassifier, Xvalidate, yValidate)

'''
In this method I calculate the best of parameters for the sklearn Pipeline. 
Then, I make predictions on the validation set and evaluate metrics and scores

Arguments:
gridSearchClassifier <=> GridSearchCV object fitted with feedback data
parameters <=> Initial parameters for Pipeline components
Xvalidate <=> Learning component of validation set
yValidate <=> Prediction component of validation set

Returns:
bestParameters <=> Best set of parameters for sklearn Pipeline after retraining with feedback data
predictions <=> Predictions on the validation set
'''
def validate(parameters, gridSearchClassifier,Xvalidate, yValidate):
    
    # Calculate best score for gridSearchClassifier
    print ('best score: %0.3f' % gridSearchClassifier.best_score_)
    # Calculate best set of parameters for gridSearchClassifier
    bestParameters = gridSearchClassifier.best_estimator_.get_params()
    # Display best set of parameters
    print ('best parameters set:')
    for paramName in sorted(parameters.keys()):
        print ('\t %s: %r' % (paramName, bestParameters[paramName]))
        
    # Make predictions on validation set and evaluate performance of gridSearchClassifier
    predictions = gridSearchClassifier.predict(Xvalidate)
    print ('Accuracy:', accuracy_score(yValidate, predictions))
    '''
    print ('Confusion Matrix:', confusion_matrix(yValidate, predictions))
    print ('Classification Report:', classification_report(yValidate, predictions))
    '''
	# Return best set of parameters and predictions
    return bestParameters, predictions
