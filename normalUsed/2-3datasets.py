from __future__ import print_function
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


if __name__ == "__main__":
  #load boston dataset
  loaded_data = datasets.load_boston()
  data_X = loaded_data.data
  data_y = loaded_data.target
  #load LinearRegression model, 
  # and the model use fit() function to fit datasets.  
  model = LinearRegression()
  model.fit( data_X, data_y )

  print(model.predict(data_X[:4,:]))
  print(data_y[:4])

  #use visitable tool to show data, 
  # and change some attribute to show the difference.
  X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)
  plt.scatter(X, y)
  plt.show()


