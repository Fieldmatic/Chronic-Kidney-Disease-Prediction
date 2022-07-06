from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

def decision_tree_classify(X_train, Y_train, X_test, Y_test):
    dtc = DecisionTreeClassifier()
    grid_param = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, 10],
        'splitter': ['best', 'random'],
        'min_samples_leaf': [1, 2, 3, 5, 7],
        'min_samples_split': [1, 2, 3, 5, 7],
        'max_features': ['1.0', 'sqrt', 'log2']
    }

    grid_search_dtc = GridSearchCV(dtc, grid_param, cv=5, n_jobs=-1, verbose=1)
    grid_search_dtc.fit(X_train, Y_train)
    dtc = grid_search_dtc.best_estimator_

    # accuracy score, confusion matrix and classification report of decision tree

    dtc_acc = accuracy_score(Y_test, dtc.predict(X_test))

    print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(Y_train, dtc.predict(X_train))}")
    print(f"Test Accuracy of Decision Tree Classifier is {dtc_acc} \n")

    print(f"Confusion Matrix :- \n{confusion_matrix(Y_test, dtc.predict(X_test))}\n")
    print(f"Classification Report :- \n {classification_report(Y_test, dtc.predict(X_test))}")

    return dtc_acc

