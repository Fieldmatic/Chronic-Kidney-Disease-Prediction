from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier


def knn_classify(X_train, Y_train, X_test, Y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    knn_acc = accuracy_score(Y_test, knn.predict(X_test))

    print(f"Training Accuracy of KNN is {accuracy_score(Y_train, knn.predict(X_train))}")
    print(f"Test Accuracy of KNN is {knn_acc} \n")

    print(f"Confusion Matrix :- \n{confusion_matrix(Y_test, knn.predict(X_test))}\n")
    print(f"Classification Report :- \n {classification_report(Y_test, knn.predict(X_test))}")