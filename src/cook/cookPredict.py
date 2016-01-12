'''
Created on Dec 4, 2015

@author: atomar
'''
import pandas as pd
import pickle
import re
from sklearn.cross_validation import train_test_split
from sklearn.metrics.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

'''
This method creates an optimal sklearn Pipeline as per the best set of
parameters obtained in cookTrain.py
'''
def getPipeline():
	
    # Load best set of parameters
    bestParameters = pickle.load(open("bestParams.pkl","rb"))
    # Create sklearn Pipeline
    pip = Pipeline([
    ('vect', TfidfVectorizer(
                             stop_words='english',
                             sublinear_tf=True,
                             use_idf=bestParameters['vect__use_idf'],
                             max_df=bestParameters['vect__max_df'],
                             ngram_range=bestParameters['vect__ngram_range']
                             )),         
                               
    ('clf', LogisticRegression(C=bestParameters['clf__C']))
    ])
    # We create this empty dict as it is required for the syntax of GridSearchCV
    parameters = {}
    # Return sklearn Pipeline and empty dict
    return pip, parameters

def main():
   
    # Get optimal sklearn Pipeline
    pip, parameters = getPipeline()
    # Create gridSearchClassifier from optimal Pipeline
    gridSearchClassifier = GridSearchCV(pip,parameters,n_jobs=3, verbose=1, scoring='accuracy')
    # Load Test Set
    testdf = pd.read_json("test.json") 
    # Remove everything but alphabets and then Lemmatize. Also, remove extra whitespace
    testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       
    # Predict Cuisine on Test Set
    predictions=gridSearchClassifier.predict(testdf['ingredients_string'])
    # Create new column in Test dataframe
    testdf['cuisine'] = predictions
    # Save the dataframe with the new column
    testdf.to_csv("submission.csv",index=False,columns=['id','cuisine'])
    
if __name__ == '__main__':
    main()
