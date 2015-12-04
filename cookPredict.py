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

def main():
   
    bestParameters = pickle.load(open("../../picks/bestParams.pkl","rb"))
    
    traindf = pd.read_json('../../data/train.json')
    
    traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

    X, y = traindf['ingredients_string'], traindf['cuisine'].as_matrix()
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, trainSize=0.7)
    
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
    
    parameters = {}
    
    gridSearchTS = GridSearchCV(pip,parameters,n_jobs=3, verbose=1, scoring='accuracy')
    gridSearchTS.fit(Xtrain, ytrain)
    
    predictions = gridSearchTS.predict(Xtest)
    
    print ('Accuracy:', accuracy_score(ytest, predictions))
    print ('Confusion Matrix:', confusion_matrix(ytest, predictions))
    print ('Classification Report:', classification_report(ytest, predictions))
    
    testdf = pd.read_json("../../data/test.json") 
   
    testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       

    predictions=gridSearchTS.predict(testdf['ingredients_string'])
    
    testdf['cuisine'] = predictions
    
    print(testdf.info())
    
    testdf.to_csv("submission.csv",index=False,columns=['id','cuisine'])
    
if __name__ == '__main__':
    main()
