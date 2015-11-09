def main():

#creating a custom pipeline

    pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english',sublinear_tf=True)),
    ('clf', LogisticRegression())
    ])

#setting the possibilities for parameters. i have found this extremely useful

    parameters = {
        'vect__max_df': (0.25, 0.5, 0.6, 0.7, 1.0),
        'vect__ngram_range': ((1, 1), (1, 2), (2,3), (1,3), (1,4), (1,5)),
        'vect__use_idf': (True, False),
        'clf__C': (0.1, 1, 10, 20, 30)
    }

#preparing the training data

    traindf = pd.read_json('../../data/train.json')

    traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]  

    traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

    X, y = traindf['ingredients_string'], traindf['cuisine'].as_matrix()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    
    
    #initializing GridSearchCV to try out all the options that i have provided above

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

#now i print the best score. this is super useful!

    print ('best score: %0.3f' % grid_search.best_score_)
    print ('best parameters set:')

#not only that, i can also get the best parameters in a nice map.

    bestParameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print ('\t %s: %r' % (param_name, bestParameters[param_name]))
        res.write('\t %s: %r\n' % (param_name, bestParameters[param_name]))

#and finally i can predict

    predictions = grid_search.predict(X_test)
    print ('Accuracy:', accuracy_score(y_test, predictions))
    print ('Confusion Matrix:', confusion_matrix(y_test, predictions))
    print ('Classification Report:', classification_report(y_test, predictions))
