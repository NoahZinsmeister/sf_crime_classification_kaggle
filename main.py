import os
import pickle
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import label_binarize

if __name__ == "__main__":
    # create a matrix of features (X) and a vector of class labels (y)
    X = []
    y = []
    with open(os.getcwd() + '/train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            # skip header
            if i == 0:
                pass
            else:
                date = re.search("([0-9]{4})-([0-9]{2})-([0-9]{2})",
                                 row[0]).groups()
                # date is of the form [year, month, day]
                date = [int(x) for x in date]
                time = re.search("([0-9]{2}):([0-9]{2}):([0-9]{2})",
                                 row[0]).groups()
                # time is of the form [hour, minute, second]
                time = [int(x) for x in time]
                category_string = row[1]
                dayofweek_string = row[3]
                pddistrict_string = row[4]
                longitude = float(row[7])
                latitude = float(row[8])
                X_row = date + time + [longitude, latitude, \
                    dayofweek_string, pddistrict_string]
                y_label = category_string
                X.append(X_row)
                y.append(y_label)

    # one-hot encoding for dayofweek and pddistrict vars
    dayofweek_set = set()
    pddistrict_set = set()
    for row in X:
        dayofweek_set.add(row[-2])
        pddistrict_set.add(row[-1])
    dayofweek_dict = {item: i for i, item in enumerate(dayofweek_set)}
    pddistrict_dict = {item: i for i, item in enumerate(pddistrict_set)}
    num_unique_dayofweek = len(dayofweek_dict)
    num_unique_pddistrict = len(pddistrict_dict)
    for i, row in enumerate(X):
        encoded_dayofweek = [0]*num_unique_dayofweek
        encoded_pddistrict = [0]*num_unique_pddistrict
        current_dayofweek = row[-2]
        current_pddistrict = row[-1]
        encoded_dayofweek[dayofweek_dict[current_dayofweek]] = 1
        encoded_pddistrict[pddistrict_dict[current_pddistrict]] = 1
        X[i] = row[:-2] + encoded_dayofweek + encoded_pddistrict

    # label binarization
    category_set = set()
    for label in y:
        category_set.add(label)
    category_dict = {item: i for i, item in enumerate(sorted(category_set))}
    num_unique_category = len(category_dict)
    for i, label in enumerate(y):
        y[i] = category_dict[label]
    #y = label_binarize(y, classes = list(range(num_unique_category)))

    # ranges for cross validation parameters
    #n_estimators_range = [i for i in range(10,331,40)]
    #max_features_range = [i for i in range(2,11,2)]
    n_estimators_range = [i for i in range(20,22,40)]
    max_features_range = [i for i in range(3,5,2)]

    # does CV and pickles the final model trained with best parameters
    param_grid = {'n_estimators': n_estimators_range, 'max_features':
                  max_features_range}
    rfc = RandomForestClassifier(random_state = 2, n_jobs = -1)
    clf = GridSearchCV(rfc, param_grid = param_grid, scoring = make_scorer(log_loss, greater_is_better = False, needs_proba = True),
                       refit = True, cv = 4)
    trained_clf = clf.fit(X, y)

    # CV plot
    scores = [-1*x[1] for x in trained_clf.grid_scores_]
    scores = np.array(scores).reshape(len(max_features_range), 
                                      len(n_estimators_range))
    plt.figure(figsize=(12, 12), dpi = 400)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xlabel('n_estimators')
    plt.ylabel('max_features')
    plt.colorbar()
    plt.xticks(np.arange(len(n_estimators_range)), n_estimators_range, rotation=0)
    plt.yticks(np.arange(len(max_features_range)), max_features_range)
    plt.figtext(.5,.96,'4-Fold CV Accuracy', fontsize = 25, ha = 'center')
    plt.figtext(.5,.94, "Best Performance:" + str(-1*trained_clf.best_score_), fontsize = 15, ha = 'center')
    plt.figtext(.5,.92, "Best Parameters:" + str(trained_clf.best_params_), fontsize = 15, ha = 'center')
    plt.savefig("CV_plot.png")

    #pickle.dump(trained_clf.best_estimator_, open("trainedclassifier.p", "wb"))

    X_test = []
    with open(os.getcwd() + '/test.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            # skip header
            if i == 0:
                pass
            else:
                date = re.search("([0-9]{4})-([0-9]{2})-([0-9]{2})", row[1]).groups()
                # date is of the form [year, month, day]
                date = [int(x) for x in date]
                time = re.search("([0-9]{2}):([0-9]{2}):([0-9]{2})", row[1]).groups()
                # time is of the form [hour, minute, second]
                time = [int(x) for x in time]
                dayofweek_string = row[2]
                pddistrict_string = row[3]
                longitude = float(row[5])
                latitude = float(row[6])
                X_row = date + time + [longitude, latitude, \
                    dayofweek_string, pddistrict_string]
                X_test.append(X_row)

    # one-hot encoding for dayofweek and pddistrict vars from existing dicts
    for i, row in enumerate(X_test):
        encoded_dayofweek = [0]*num_unique_dayofweek
        encoded_pddistrict = [0]*num_unique_pddistrict
        current_dayofweek = row[-2]
        current_pddistrict = row[-1]
        try:
            encoded_dayofweek[dayofweek_dict[current_dayofweek]] = 1
        except KeyError:
            encoded_dayofweek[0] = 1
        try:
            encoded_pddistrict[pddistrict_dict[current_pddistrict]] = 1
        except KeyError:
            encoded_pddistrict[0] = 1
        X_test[i] = row[:-2] + encoded_dayofweek + encoded_pddistrict

    num_classes = trained_clf.best_estimator_.n_classes_
    predicted_probas = trained_clf.predict_proba(X_test)
    with open(os.getcwd() + '/submit.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Id"] + sorted(category_set))
        for i, prediction in enumerate(predicted_probas):
            writer.writerow([i] + prediction.tolist())
