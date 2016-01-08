'''
Created on Dec 31, 2015

@author: abtpst
'''
import pandas as pd
import cookUtil as cookut
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.cross_validation import train_test_split

def feedback(pipeline,parameters,ultimateTraindf):
    
    fbkdf = pd.read_csv("../../results/feedback.csv")
    
    fbkdf = fbkdf.loc[fbkdf['check'] == False]
    fbkdf = fbkdf.append(ultimateTraindf, ignore_index=True)
    
    X, y = fbkdf['properIngredients'], fbkdf['cuisine'].as_matrix()
    
    Xtrain, Xvalidate, ytrain, yValidate = train_test_split(X, y, train_size=0.7)
    # Initialize gridSearchClassifierCV Classifier with parameters
    gridSearchClassifier = cookut.getGridSearchCv(pipeline, parameters)
    # Fit transform the gridSearchClassifier on Training Set
    gridSearchClassifier.fit(Xtrain,ytrain)
    
    return validate(parameters, gridSearchClassifier, Xvalidate, yValidate)

def validate(parameters, gridSearchClassifier,Xvalidate, yValidate):
    
    # Calculate best score for gridSearchClassifier
    print ('best score: %0.3f' % gridSearchClassifier.best_score_)
    # Calculate best set of parameters for gridSearchClassifier
    bestParameters = gridSearchClassifier.best_estimator_.get_params()
    # Display best set of parameters
    print ('best parameters set:')
    for paramName in sorted(parameters.keys()):
        print ('\t %s: %r' % (paramName, bestParameters[paramName]))
        
    # Evaluate performance of gridSearchClassifier on Validation Set
    predictions = gridSearchClassifier.predict(Xvalidate)
    print ('Accuracy:', accuracy_score(yValidate, predictions))
    '''
    print ('Confusion Matrix:', confusion_matrix(yValidate, predictions))
    print ('Classification Report:', classification_report(yValidate, predictions))
    '''
    return bestParameters, predictions
