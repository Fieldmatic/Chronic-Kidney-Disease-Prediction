from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

def decision_tree_classify(X_train, Y_train, X_test, Y_test):
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, Y_train)
    dtc_acc = accuracy_score(Y_test, dtc.predict(X_test))

    print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(Y_train, dtc.predict(X_train))}")
    print(f"Test Accuracy of Decision Tree Classifier is {dtc_acc} \n")

    print(f"Confusion Matrix :- \n{confusion_matrix(Y_test, dtc.predict(X_test))}\n")
    print(f"Classification Report :- \n {classification_report(Y_test, dtc.predict(X_test))}")
