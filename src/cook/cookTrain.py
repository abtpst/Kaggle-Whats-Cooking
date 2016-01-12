import pandas as pd
import re
import pickle
import cookValidate as cookVal
import cookUtil as cookut
from nltk.stem import WordNetLemmatizer
from sklearn.cross_validation import train_test_split
'''
This method calculates and displays the best set of parameters for the sklearn Pipeline
It also shows the accuracy of the classifier on the validation set

Arguments:
pipeline <=> sklearn Pipeline with TfidfVectorizer and LogisticRegression
parameters <=> Parameters for initializing Pipeline components

Returns:
bestParameters <=> Best set of parameters for sklearn Pipeline after retraining with feedback data
'''
def getBestParameters(pipeline,parameters):

    # Load training data
    traindf = pd.read_json('../../data/train.json')
    # Remove everything but alphabets and then Lemmatize. Also remove extra whitespace
    traindf['properIngredients'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       
    # Create learning matrix
    X, y = traindf['properIngredients'], traindf['cuisine'].as_matrix()
    # Split into Training and Validation Sets
    Xtrain, Xvalidate, ytrain, yValidate = train_test_split(X, y, train_size=0.7)
    # Initialize gridSearchClassifierCV Classifier with parameters
    gridSearchClassifier = cookut.getGridSearchCv(pipeline, parameters)
    # Fit/train the gridSearchClassifier on Training Set
    gridSearchClassifier.fit(Xtrain, ytrain)
    # Make predictions on validation set and calculate best set of parameters  
    bestParameters,predictions=cookVal.validate(parameters, gridSearchClassifier, Xvalidate, yValidate)
    # Initialize DataFrame for feedback loop
    valdf = pd.DataFrame(index = Xvalidate.index.values)
    # Add ingredients column
	valdf=valdf.join(Xvalidate)
	# Add correct cuisine
    valdf["cuisine"] = yValidate
	# Add predictions column
    valdf["pred_cuisine"] = predictions
    # Add check column. This column would be false for incorrect predictions
	valdf["check"] = valdf.pred_cuisine==valdf.cuisine
    # Store DataFrame for feedback
    valdf.to_csv("../../results/feedback.csv")
    # Create joint DataFrame to incorporate feedback data. As of now, this will only have the ingredients and cuisine columns from the training set
    ultimateTraindf = pd.DataFrame(index = Xtrain.index.values)
    ultimateTraindf=ultimateTraindf.join(Xtrain)
    ultimateTraindf["cuisine"] = ytrain
    # Calculate best set of parameters after retraining with feedback data. Make predictions on validation set 
    bestParameters,predictions = cookVal.feedback(pipeline,parameters,ultimateTraindf)
    # Return best set of parameters
    return bestParameters
    
def main():
    
    # Get Pipeline components
    pipeline = cookut.getPipeline()
    
    # Get parameter options for Pipeline components
    parameters = cookut.getParameters()
    
    # Get best set of parameters and evaluate validation set accuracy
    bestParameters = getBestParameters(pipeline,parameters)

    # Save best parameter set
    res = open("../../picks/res.txt", 'w')
    res.write ('best parameters set:\n')
    
    for paramName in sorted(parameters.keys()):
        res.write('\t %s: %r\n' % (paramName, bestParameters[paramName]))
    
    pickle.dump(bestParameters,open("../../picks/bestParams.pkl","wb"))
    
if __name__ == '__main__':
    main()
