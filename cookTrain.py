import pandas as pd
import re
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.gridSearch import GridSearchCV

from IPython.display import Image

def main():
    pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english',sublinear_tf=True)),
    ('clf', LogisticRegression())
    ])
    
    parameters = {
        'vect__max_df': (0.25, 0.5, 0.6, 0.7, 1.0),
        'vect__ngram_range': ((1, 1), (1, 2), (2,3), (1,3), (1,4), (1,5)),
        'vect__use_idf': (True, False),
        'clf__C': (0.1, 1, 10, 20, 30)
    }
    
    traindf = pd.read_json('../../data/train.json')
    
    traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]  

    traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

    X, y = traindf['ingredients_string'], traindf['cuisine'].as_matrix()
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, trainSize=0.7)
    
    gridSearch = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')
    gridSearch.fit(Xtrain, ytrain)
    
    print ('best score: %0.3f' % gridSearch.best_score_)
    print ('best parameters set:')
    
    res = open("../../picks/res.txt", 'w')
    res.write ('best parameters set:\n')
    bestParameters = gridSearch.best_estimator_.get_params()
    for paramName in sorted(parameters.keys()):
        print ('\t %s: %r' % (paramName, bestParameters[paramName]))
        res.write('\t %s: %r\n' % (paramName, bestParameters[paramName]))
    
    pickle.dump(bestParameters,open("../../picks/bestParams.pkl","wb"))
    
    predictions = gridSearch.predict(Xtest)
    print ('Accuracy:', accuracy_score(ytest, predictions))
    print ('Confusion Matrix:', confusion_matrix(ytest, predictions))
    print ('Classification Report:', classification_report(ytest, predictions))
    
if __name__ == '__main__':
    main()
