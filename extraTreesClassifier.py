from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def extra_trees_classify(X_train, Y_train, X_test, Y_test):
    etc = ExtraTreesClassifier()
    etc.fit(X_train, Y_train)

    # accuracy score, confusion matrix and classification report of extra trees classifier

    etc_acc = accuracy_score(Y_test, etc.predict(X_test))

    print(f"Training Accuracy of Extra Trees Classifier is {accuracy_score(Y_train, etc.predict(X_train))}")
    print(f"Test Accuracy of Extra Trees Classifier is {etc_acc} \n")

    print(f"Confusion Matrix :- \n{confusion_matrix(Y_test, etc.predict(X_test))}\n")
    print(f"Classification Report :- \n {classification_report(Y_test, etc.predict(X_test))}")