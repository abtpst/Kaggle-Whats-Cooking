import pandas as pd
import re
import pickle
import cookValidate as cookVal
import cookUtil as cookut
from nltk.stem import WordNetLemmatizer
from sklearn.cross_validation import train_test_split

# This method calculates and displays the best set of parameters for the sklearn Pipeline
# It also shows the accuracy of the classifier on the validation set
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
  
    bestParameters,predictions=cookVal.validate(parameters, gridSearchClassifier, Xvalidate, yValidate)
    
    valdf = pd.DataFrame(index = Xvalidate.index.values)
    valdf=valdf.join(Xvalidate)
    valdf["cuisine"] = yValidate
    valdf["pred_cuisine"] = predictions
    valdf["check"] = valdf.pred_cuisine==valdf.cuisine
    
    valdf.to_csv("../../results/feedback.csv")
    
    ultimateTraindf = pd.DataFrame(index = Xtrain.index.values)
    ultimateTraindf=ultimateTraindf.join(Xtrain)
    ultimateTraindf["cuisine"] = ytrain
    
    bestParameters,predictions = cookVal.feedback(pipeline,parameters,ultimateTraindf)
    
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
