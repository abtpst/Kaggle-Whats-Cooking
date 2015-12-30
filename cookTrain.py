import pandas as pd
import re
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.gridSearchClassifier import gridSearchClassifierCV

from IPython.display import Image

# This method defines the sklearn Pipeline. It can be modified to change the Pipeline components.
# For simplicity, I am just returning a hard-coded value.
def getPipeline():

	return Pipeline([
    ('vect', TfidfVectorizer(stop_words='english',sublinear_tf=True)),
    ('clf', LogisticRegression())
    ])

# This method specifies the parameter options for sklearn Pipeline. It can be modified to change the parameter options.
# For simplicity, I am just returning a hard-coded value.
def getParameters():

	return {
        'vect__max_df': (0.25, 0.5, 0.6, 0.7, 1.0),
        'vect__ngram_range': ((1, 1), (1, 2), (2,3), (1,3), (1,4), (1,5)),
        'vect__use_idf': (True, False),
        'clf__C': (0.1, 1, 10, 20, 30)
    }

# This method calculates and displays the best set of parameters for the sklearn Pipeline
# It also shows the accuracy of the classifier on the validation set
def getBestParameters(pipeline,parameters):
	
    # Load training data
    traindf = pd.read_json('../../data/train.json')
    # Remove extra whitespace
    traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]  
    # Remove everything but alphabets and then Lemmatize
    traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       
    # Create learning matrix
    X, y = traindf['ingredients_string'], traindf['cuisine'].as_matrix()
    # Split into Training and Validation Sets
    Xtrain, Xvalidate, ytrain, yValidate = train_test_split(X, y, trainSize=0.7)
    # Initialize gridSearchClassifierCV Classifier with parameters
    gridSearchClassifier = gridSearchClassifierCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')
    # Fit/train the gridSearchClassifier on Training Set
	gridSearchClassifier.fit(Xtrain, ytrain)
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
    print ('Confusion Matrix:', confusion_matrix(yValidate, predictions))
    print ('Classification Report:', classification_report(yValidate, predictions))
	
    return bestParameters
	
def main():
    
	# Get Pipeline components
    pipeline = getPipeline()
    
    # Get parameter options for Pipeline components
    parameters = getParameters()
    
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
