'''
Created on Dec 31, 2015

@author: abtpst
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

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
        'vect__max_df': [0.25, 0.5, 0.6, 0.7, 1.0],
        'vect__ngram_range': [(1, 1), (1, 2), (2,3), (1,3), (1,4), (1,5)],
        'vect__use_idf': [True, False],
        'clf__C': [0.1, 1, 10, 20, 30]
    }

def getGridSearchCv(pipeline,parameters):
    
    return GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')