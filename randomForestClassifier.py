from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def random_forest_classify(X_train, Y_train, X_test, Y_test):
    rd_clf = RandomForestClassifier(criterion='entropy', max_depth=11, max_features='auto', min_samples_leaf=2,
                                    min_samples_split=3, n_estimators=130)
    rd_clf.fit(X_train, Y_train)

    # accuracy score, confusion matrix and classification report of random forest

    rd_clf_acc = accuracy_score(Y_test, rd_clf.predict(X_test))

    print(f"Training Accuracy of Random Forest Classifier is {accuracy_score(Y_train, rd_clf.predict(X_train))}")
    print(f"Test Accuracy of Random Forest Classifier is {rd_clf_acc} \n")

    print(f"Confusion Matrix :- \n{confusion_matrix(Y_test, rd_clf.predict(X_test))}\n")
    print(f"Classification Report :- \n {classification_report(Y_test, rd_clf.predict(X_test))}")

    return rd_clf_acc
