def main():

    # just load the best parameters from the training script
    
    bestParameters = pickle.load(open("../../picks/bestParams.pkl","rb"))
    
    # create optimal pipeline
    
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
    
    # still need to have a parameters map due to syntax. so just leave it empy this time
    
    parameters = {}
    
    # prepare training data so that the pipeline can be trained
    
    traindf = pd.read_json('../../data/train.json')
    
    traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

    X, y = traindf['ingredients_string'], traindf['cuisine'].as_matrix()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    
    # train optimal pipeline
    
    grid_searchTS = GridSearchCV(pip,parameters,n_jobs=3, verbose=1, scoring='accuracy')
    grid_searchTS.fit(X_train, y_train)
    
    # no need to do this, but still can be used to ensure that the pipeline we created is indeed the optimal one
    
    predictions = grid_searchTS.predict(X_test)
    
    print ('Accuracy:', accuracy_score(y_test, predictions))
    print ('Confusion Matrix:', confusion_matrix(y_test, predictions))
    print ('Classification Report:', classification_report(y_test, predictions))
    
    # now prepare the test data
    
    testdf = pd.read_json("../../data/test.json") 
   
    testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       

    predictions=grid_searchTS.predict(testdf['ingredients_string'])
    
    add a new column and populate as per the predictions we got
    
    testdf['cuisine'] = predictions
    
    print(testdf.info())
    
    # save as csv and submit to Kaggle
    testdf.to_csv("submission.csv",index=False,columns=['id','cuisine'])
