import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


if __name__ == "__main__":
    #load data from iris database
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target
    print(iris_X[:2,:])
    print(iris_y)
    print('---------------------------------')

    #The data devide to testing and traning data, and the testing data is 30% in the data.
    #shuffle data is important in ML.
    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3 )
    print(y_train)
    print('---------------------------------')

    #crate model, testing and training
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    print(knn.predict(X_test))
    print(y_test)