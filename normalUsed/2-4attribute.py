from sklearn import datasets
from sklearn.linear_model import LinearRegression

if __name__=="__main__":
  #load dataset
  loaded_data = datasets.load_boston() 
  data_X = loaded_data.data
  data_y = loaded_data.target
  #import model, train and predict
  model = LinearRegression()
  model.fit(data_X, data_y)
  print(model.predict(data_X[:4,:]))
  print("-----------------------------------")
  print(model.coef_)
  print(model.intercept_)
  print("-----------------------------------")
  print(model.get_params())  
  print("-----------------------------------")
  print(model.score(data_X, data_y)) # R^2 coefficient of determination