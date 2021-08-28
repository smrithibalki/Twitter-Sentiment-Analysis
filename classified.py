#!/usr/bin/env python
# coding: utf-8

# !/usr/bin/env python
# coding: utf-8

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import naive_bayes, tree, ensemble, svm, linear_model, neural_network
from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle  # Edited
from xgboost import XGBClassifier

folder_path = ''
test = True

# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1,1e-1, 1e-2, 1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [9.1, 1, 10, 100, 1000]}]
models = {
    #     'XGBoost': XGBClassifier(),
    #     'MNB': naive_bayes.MultinomialNB(),
    #     'Dec_Tree': tree.DecisionTreeClassifier(random_state=0),
    #      'RdmForest': ensemble.RandomForestClassifier(criterion='entropy', n_jobs = 10),
          'LR':linear_model.LogisticRegression(solver='lbfgs', max_iter=1000),
    #      'MLP':neural_network.MLPClassifier(hidden_layer_sizes=(50,50,50,50), learning_rate='constant',
    #               learning_rate_init=0.001, max_fun=250000, max_iter=1000,),
    #      'RdmForest': ensemble.RandomForestClassifier(n_estimators=300,max_depth=100,max_features=15),
    #

    #      'SVM': GridSearchCV(svm.SVC(), tuned_parameters)  ,
    #      'SVM' : svm.SVC(),
    # 'eclf': ensemble.VotingClassifier(estimators=[  # Edited
    #
    #     ('rf', ensemble.RandomForestClassifier(n_estimators=300, max_depth=100, max_features=15)),
    #     ('lr', linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)),
    #     ('mnb', naive_bayes.MultinomialNB()),
    #     #              ('svm', svm.SVC(probability=True)),
    #     #              ('mlp',neural_network.MLPClassifier(hidden_layer_sizes=(50,50,50,50), learning_rate='constant',
    #     #               learning_rate_init=0.001, max_fun=250000, max_iter=1000,)),
    #     #              ('xgb', XGBClassifier()),
    #
    # ], voting='soft')
}


def vectorizer(candidate, train_tweets, test_tweets):
    if test:
        with open(candidate + "vect.pickle", "rb") as inp_vect:  # Edited
            vectorizer = pickle.load(inp_vect)
        train_vectors = vectorizer.transform(train_tweets)
        test_vectors = vectorizer.transform(test_tweets)
        return train_vectors, test_vectors
    else:
        vectorizer = text.TfidfVectorizer(
            min_df=0.00125,
            max_df=0.7,
            sublinear_tf=True,
            use_idf=True,
            analyzer='word',
            ngram_range=(1, 5)
        )
        train_vectors = vectorizer.fit_transform(train_tweets)
        with open(candidate + "vect.pickle", "wb") as op_vect:  # Edited
            vectorizer = pickle.dump(vectorizer, op_vect)
        return train_vectors, []


def feature_generation(candidate, train_file, test_file=''):
    train_data = pd.read_csv(train_file)
    train_tweets = train_data['tweet'].astype(str)
    train_class = train_data['label']

    test_tweets = []
    if test_file:
        test_data = pd.read_csv(test_file)
        test_tweets = test_data['tweet'].astype(str)

    train_vectors, test_vectors = vectorizer(candidate, train_tweets, test_tweets)
    return train_vectors, train_class, test_vectors, test_tweets


def over_sample(train_vectors, train_class):
    train_vectors = train_vectors.toarray()
    sm = SMOTE(random_state=42)
    train_vectors, train_class = sm.fit_resample(train_vectors, train_class)
    return train_vectors, train_class


def classify(candidate, model, classifier, train_vectors, train_class, test_vectors):  # Edited
    if test:
        '''Fit the model and predict the labels'''
        # with open(r"trained_models/main_" + candidate + model + ".pickle", "rb") as input_file: # Edited
        with open(r"trained_models/" + candidate + model + ".pickle", "rb") as input_file:
            print(input_file)
            classifier = pickle.load(input_file)
        predictions = classifier.predict(test_vectors)
        #print(predictions)
        return predictions

    else:
        ''' If not a test file,
            cross validate using classifier and evaluate metrics
        '''
        # with open(r"trained_models/main_" + candidate + model + ".pickle", "wb") as output_file: # Edited
        with open(r"trained_models/" + candidate + model + ".pickle", "wb") as output_file:  # Edited
            model_obj = classifier.fit(train_vectors, np.ravel(train_class))
            pickle.dump(model_obj, output_file)
        # if model != 'eclf':
        #         predictions = cross_val_predict(classifier, train_vectors, train_class, cv=5) # Replace with more efficient fn?
        #         print("Classified using: ",classifier)
        #         accuracy = accuracy_score(train_class,predictions)
        #         labels = [1,-1, 0]
        #         precision = precision_score(train_class, predictions, average=None,labels=labels)
        #         recall = recall_score(train_class,predictions,average=None,labels=labels)
        #         f1score = f1_score(train_class,predictions,average=None,labels=labels)
        #         print("accuracy", accuracy)
        #         print(precision,recall,f1score)
        #         return accuracy, precision, recall, f1score
        #         else:
        return -1, -1, -1, [-1, -1]


def train_classify(candidate, train_file, test_file):
    text = ""

    ''' Step-1 Feature Generation'''
    train_vectors, train_class, test_vectors, test_tweets = feature_generation(candidate, train_file, test_file)

    if not test:
        ''' Step-2 Over sampling so that all classes have equal probability distribution'''
        train_vectors, train_class = over_sample(train_vectors, train_class)

    if test:
        print(candidate)
        for i, model in enumerate(models):
            predictions = classify(candidate, model, "", train_vectors, train_class, test_vectors.toarray())  # Edited

            # accuracy = accuracy_score(train_class, predictions)
            # labels = [1, -1, 0]
            # precision = precision_score(train_class, predictions, average=None, labels=labels)
            # recall = recall_score(train_class, predictions, average=None, labels=labels)
            # f1score = f1_score(train_class, predictions, average=None, labels=labels)
            # print(model + "\n Accuracy : " + str(round(accuracy, 2) * 100) + "%\n Positive F1 Score : " + str(
            #     round(f1score[0], 2) * 100) + "%\n Negative F1 Score : " + str(round(f1score[1], 2) * 100) +
            #       "%\n Neutral F1 Score : " + str(round(f1score[2], 2) * 100) + "%\n Positive precision score : "+str(
            #     round(precision[0],2) * 100)+"%\n Negative precision Score : " + str(round(precision[1], 2) * 100)+
            #       "%\n Neutral precision Score : " + str(round(precision[2], 2) * 100)+"%\n Positive recall Score : " + str(
            #     round(recall[0], 2) * 100) +"%\n Negative recall Score : " + str(round(recall[1], 2) * 100) +
            #       "%\n Neutral recall Score : " + str(round(recall[2], 2) * 100))
            '''Store the predicted classes labels in txt file'''
            f = open("output/" + candidate + '.txt', 'w+')
            f.write("79-70")
            f.write("\n")
            for index, pred in enumerate(predictions):

                f.write(str(index + 1) + ';;' + str(predictions[index]) + '\n')
            f.close()
    else:
        '''Evaluate the training data'''
        metrics = []
        for i, model in enumerate(models):
            accuracy, precision, recall, f1score = classify(candidate, model, models[model], train_vectors, train_class,
                                                            test_vectors)

            metrics.append({})
            metrics[i]['Classifier'] = model
            metrics[i]['accuracy'] = accuracy
            metrics[i]['positive f1score'] = f1score[0]
            metrics[i]['negative f1score'] = f1score[1]

            ''' Store the evaluation metrics in a text file'''

            text += model + "\n Accuracy : " + str(round(accuracy, 2) * 100) + "%\n Positive F1 Score : " + str(
                round(f1score[0], 2) * 100) + "%\n Negative F1 Score : " + str(round(f1score[1], 2) * 100) + "%\n"

        f = open(folder_path + candidate + "_output_train.txt", "w")  # Edited
        f.write(text)
        f.close()
        print(candidate, "done")


if test:
    '''For test data we take the cleaned test data'''
    for candidate in ['obama', 'romney']:
        print('Running for', candidate, '...')
        test_file = folder_path + candidate + '_test_cleaned.csv' if test else ''  # Edited
        train_classify(candidate, folder_path +candidate+'_train_cleaned.csv', test_file)
else:
    '''For training data- trained cleaned data is taken with empty test file'''
    for candidate in ['obama', 'romney']:
        print('Running for', candidate, '...')
        test_file = folder_path + candidate + '_train_cleaned.csv' if test else '' # Edited
        train_classify(candidate, folder_path +candidate+'_train_cleaned.csv', test_file) # Edited







