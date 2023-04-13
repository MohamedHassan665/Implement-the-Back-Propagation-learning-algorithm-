from builtins import print

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
def samplePreprocessing(sampleData, minMaxDF):
  if "gender" in sampleData.columns:
    if sampleData.at[0,"gender"] == "male":
      sampleData.at[0,"gender"] = 0
    elif sampleData.at[0,"gender"] == "female":
      sampleData.at[0,"gender"] = 1
  for col in sampleData.columns:
    sampleData[col] = (sampleData[col] - minMaxDF[col][0]) / (minMaxDF[col][1] - minMaxDF[col][0])
    #print(col, " ", sampleData[col], " - ", minMaxDF[col][0], " / ", minMaxDF[col][1], " - ", minMaxDF[col][0])
  return sampleData

def visualization(data):
  for col in data.columns:
    if col == "species":
      continue
    flag = 0
    for col2 in data.columns:
      if col == col2:
        flag = 1
        continue
      if flag == 1:
        colors = ['red','green','blue']
        plt.scatter(data[col], data[col2], c=data["species"], cmap=matplotlib.colors.ListedColormap(colors))
        plt.xlabel(col)
        plt.ylabel(col2)
        plt.show()

def preprocessing(data):
  maleCnt = 0
  femaleCnt = 0
  # Change Class To Neumirc
  for idx,row in data.iterrows():
    if data.at[idx, "species"] == "Adelie":
      data.at[idx, "species"] = 0
    elif data.at[idx, "species"] == "Gentoo":
      data.at[idx, "species"] = 1
    elif data.at[idx, "species"] == "Chinstrap":
      data.at[idx, "species"] = 2
    if data.at[idx,"gender"] == "male":
      data.at[idx,"gender"] = 0
      maleCnt+=1
    elif data.at[idx,"gender"] == "female":
      femaleCnt+=1
      data.at[idx,"gender"] = 1

  # Replace Null Value
  if maleCnt >= femaleCnt:
    data["gender"].fillna(0, inplace=True)
  else:
    data["gender"].fillna(1, inplace=True)
  #print(data.isnull().sum())
  minMaxDF = pd.DataFrame(columns = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "gender", "body_mass_g"])
  minMaxDF = minMaxDF.append({'bill_length_mm':0}, ignore_index=True)
  minMaxDF = minMaxDF.append({'bill_length_mm':0}, ignore_index=True)
# Make Normalization
  for col in data.columns:
    if(col == "species"):
      continue
    maxVal = data[col].max()
    minVal = data[col].min()  
    minMaxDF[col][0] = minVal
    minMaxDF[col][1] = maxVal
    data[col] = (data[col] - minVal) / (maxVal - minVal)

  return data, minMaxDF

# data = pd.read_csv("penguins.csv")
# data, minMaxDF= preprocessing(data)
# data

def trainTestSplit(data):
  trainData = data.loc[:29]
  trainData = trainData.append(data.loc[50:79])
  trainData = trainData.append(data.loc[100:129])
  trainData = trainData.reset_index()
  trainData = trainData.drop("index", axis = 1)
  testData = data.loc[30:49]
  testData = testData.append(data.loc[80:99])
  testData = testData.append(data.loc[130:149])
  testData = testData.reset_index()
  testData = testData.drop("index", axis = 1)
  trainData = trainData.sample(frac=1, random_state=60).reset_index(drop=True)
  testData = testData.sample(frac=1, random_state=60).reset_index(drop=True)
  return trainData, testData

def init_weights(hidden_nodes):
  L1=6
  Weights = []
  for i in (hidden_nodes):
    numberW = L1 * i   # number of weights for each layer
    tmp = [] 
    while(numberW >0):
      tmp.append(random.random())
      numberW= numberW-1
    tmp = np.array(tmp)
    tmp = tmp.reshape(i, L1)
    Weights.append(tmp)
    L1 = i + 1
  return Weights

def backPropagation(idx, y_train, activation_fun, Weights, f):
  error_list = []
  target = [0,0,0]
  for i in range(len(f)):
    error = []
    if i == 0:
      if(y_train[idx] == 0):
        target[0] = 1
      elif(y_train[idx] == 1):
        target[1] = 1
      else:
        target[2] = 1
      for j in range(len(f[i])):
        if(activation_fun == "sigmoid"):
          sigma = (target[j] - f[i][j])*(f[i][j])*(1-(f[i][j]))
        else:
          sigma = (target[j] - f[i][j])*(1 + f[i][j])*(1-(f[i][j]))
        error.append(sigma)
    else:
      #3shan n7sb el sum mn 8er el bias 
      for j in range(len(f[i]) - 1):
        sigma = 0
        for k in range(len(error_list[i - 1])):
          sigma = sigma + Weights[len(Weights) - i][k][j] * error_list[i - 1][k]
        if(activation_fun == "sigmoid"):
          sigma = f[i][j] *(1-f[i][j])* sigma
        else:
          sigma = (1 + f[i][j])*(1-(f[i][j]))*sigma
        error.append(sigma)
    error_list.append(error)
  return error_list

def updateWeights(idx, Weights, eta, error_list, x, f):
  idx = 0
  for i in range(len(Weights)):
    for j in range(len(Weights[i])):
      if(i == 0):
        Weights[i][j] =  Weights[i][j] + eta * error_list[i][j] * x
      else:
        Weights[i][j] =  Weights[i][j] + eta * error_list[i][j] * f[i - 1]
  return Weights

def train(Weights,epochs, x_train, y_train, x, eta, activation_fun, bias, hidden_layers):
  for i in range(epochs):
    for idx,row in x_train.iterrows():
      f = []
      x[0] = row[0]
      x[1] = row[1]
      x[2] = row[2]
      x[3] = row[3]
      x[4] = row[4]
      net = np.dot(Weights[0], x)
      print('The Value of net is  ',net)
      if(activation_fun == "sigmoid"):
        y = 1/(1 + np.exp(-net))
      else:
        y = (1-np.exp(-net))/(1 + np.exp(-net))
      if(bias == 1):
        y = np.append(y, 1)
      else:
        y = np.append(y, 0)
      f.append(y)

      for j in range(hidden_layers):
        net = np.dot(Weights[j + 1], f[j])
        if(activation_fun == "sigmoid"):
          y = 1/(1 + np.exp(-net))
        else:
          y = (1-np.exp(-net))/(1 + np.exp(-net))
        if(j != hidden_layers - 1):
          if(bias == 1):
            y = np.append(y, 1)
          else:
            y = np.append(y, 0)
        f.append(y)
      f.reverse()
      error_list = backPropagation(idx, y_train, activation_fun, Weights, f)
      f.reverse()
      error_list.reverse()
      Weights = updateWeights(idx, Weights, eta, error_list, x, f)
    print('The Result of y is ', y)
    print('The Result of F is ', f)
    break
  return Weights

def test(bias, x_test, Weights, activation_fun, hidden_layers, y_test):
  pred = []
  AA = 0
  AG = 0
  AC = 0
  GG = 0
  GA = 0
  GC = 0
  CC = 0
  CA = 0
  CG = 0
  z = np.zeros([5,1])
  if(bias == 1):
    z = np.append(z, 1)
  else:
    z = np.append(z, 0)
  for idx,row in x_test.iterrows(): #test each sample
    f = []
    z[0] = row[0]
    z[1] = row[1]
    z[2] = row[2]
    z[3] = row[3]
    z[4] = row[4]
    net = np.dot(Weights[0], z)
    if(activation_fun == "sigmoid"):
      y = 1/(1 + np.exp(-net))
    else:
      y = (1-np.exp(-net))/(1 + np.exp(-net))
    if(bias == 1):
      y = np.append(y, 1)
    else:
      y = np.append(y, 0)
    f.append(y)
    for j in range(hidden_layers):
      net = np.dot(Weights[j + 1], f[j])
      if(activation_fun == "sigmoid"):
        y = 1/(1 + np.exp(-net))
      else:
        y = (1-np.exp(-net))/(1 + np.exp(-net))
      if(j != hidden_layers - 1):
        if(bias == 1):
          y = np.append(y, 1)
        else:
          y = np.append(y, 0)
      f.append(y)
    y = np.where(y==max(y),1,0)
    index = np.where(y==1)[0][0]
    pred.append(index)
    if(index == y_test[idx]):
      if(index == 0):
        AA+=1
      elif(index == 1):
        GG+=1
      else:
        CC+=1
    else:
      if(index == 0 and y_test[idx] == 1):
        AG +=1
      elif(index == 0 and y_test[idx] == 2):
        AC += 1
      elif(index == 1 and  y_test[idx] == 0):
        GA += 1
      elif(index == 1 and y_test[idx] == 2):
        GC += 1
      elif(index == 2 and y_test[idx] == 0):
        CA += 1
      elif(index == 2 and y_test[idx] == 1):
        CG += 1


  confusionMatrix = [
        [AA, GA, CA],
        [AG, GG, CG],
        [AC, GC, CC]
    ]
  acc = 0
  for i in range(len(pred)):
    if(pred[i]==y_test[i]):
      acc+=1
  accuracy = acc/ len(y_test) * 100
  print("The testing accuracy", accuracy)
  confusionMatrix = np.array(confusionMatrix)
  print("           Confusion matrix")
  print("             True classes")
  print("       Adelie", "Gentoo", "Chinstrap")
  #print("        tp    fp")
  print("Adelie    ", AA, "  ", GA, "  ", CA)
  #print("        fn    tn")
  print("Gentoo    ", AG, "  ", GG, "  ", CG)
  print("Chinstrap ", AC, "  ", GC, "  ", CC)

# acc = 0
# for i in range(len(pred)):
#   if(pred[i]==y_test[i]):
#     acc+=1
# accuracy = acc/ len(y_test) * 100

# print("The testing accuracy", accuracy)

def acc_train(bias, x_train, Weights, activation_fun, hidden_layers, y_train):
  pred = []
  z = np.zeros([5,1])
  if(bias == 1):
    z = np.append(z, 1)
  else:
    z = np.append(z, 0)
  for idx,row in x_train.iterrows(): #test each sample
    f = []
    z[0] = row[0]
    z[1] = row[1]
    z[2] = row[2]
    z[3] = row[3]
    z[4] = row[4]
    net = np.dot(Weights[0], z)
    if(activation_fun == "sigmoid"):
      y = 1/(1 + np.exp(-net))
    else:
      y = (1-np.exp(-net))/(1 + np.exp(-net))
    if(bias == 1):
      y = np.append(y, 1)
    else:
      y = np.append(y, 0)
    f.append(y)
    for j in range(hidden_layers):
      net = np.dot(Weights[j + 1], f[j])
      if(activation_fun == "sigmoid"):
        y = 1/(1 + np.exp(-net))
      else:
        y = (1-np.exp(-net))/(1 + np.exp(-net))
      if(j != hidden_layers - 1):
        if(bias == 1):
          y = np.append(y, 1)
        else:
          y = np.append(y, 0)
      f.append(y)
    y = np.where(y==max(y),1,0)
    index = np.where(y==1)[0][0]
    pred.append(index)
  acc = 0
  for i in range(len(pred)):
    if(pred[i]==y_train[i]):
      acc+=1
  accuracy = acc/ len(y_train) * 100
  print("The training accuracy", accuracy)

# acc = 0
# for i in range(len(pred)):
#   if(pred[i]==y_train[i]):
#     acc+=1
# accuracy = acc/ len(y_train) * 100

# print("The training accuracy", accuracy)

# sampleData = [[13.2, 4500, 2000, "male", 4000]]
# def predictSample(sampleDF, x_train, minMaxDF):
#   sampleDF = pd.DataFrame(sampleData, columns = x_train.columns)
#   sampleDF = samplePreprocessing(sampleDF, minMaxDF)

# sampleDF = samplePreprocessing(sampleDF, minMaxDF)
# sampleDF

def predictSample(sampleData, x_train, minMaxDF, bias, Weights, activation_fun, hidden_layers):
  sampleDF = pd.DataFrame(sampleData, columns = x_train.columns)
  sampleDF = samplePreprocessing(sampleDF, minMaxDF)
  f = []
  z = np.zeros([5,1])
  if(bias == 1):
    z = np.append(z, 1)
  else:
    z = np.append(z, 0)
  z[0] = sampleDF.at[0,"bill_length_mm"]
  z[1] = sampleDF.at[0,"bill_depth_mm"]
  z[2] = sampleDF.at[0,"flipper_length_mm"]
  z[3] = sampleDF.at[0,"gender"]
  z[4] = sampleDF.at[0,"body_mass_g"]
  net = np.dot(Weights[0], z)
  if(activation_fun == "sigmoid"):
    y = 1/(1 + np.exp(-net))
  else:
    y = (1-np.exp(-net))/(1 + np.exp(-net))
  if(bias == 1):
    y = np.append(y, 1)
  else:
    y = np.append(y, 0)
  f.append(y)
  for j in range(hidden_layers):
    net = np.dot(Weights[j + 1], f[j])
    if(activation_fun == "sigmoid"):
      y = 1/(1 + np.exp(-net))
    else:
      y = (1-np.exp(-net))/(1 + np.exp(-net))
    if(j != hidden_layers - 1):
      if(bias == 1):
        y = np.append(y, 1)
      else:
        y = np.append(y, 0)
    f.append(y)
  print(y, "\n")
  y = np.where(y==max(y),1,0)
  index = np.where(y==1)[0][0]
  dict = {0:'Adelie', 1: 'Gentoo', 2: 'Chinstrap'}
  print("The predicted class is: ", dict[index])

def main(hidden_layers,hidden_nodes,epochs,eta,activation_fun,bias):
  data = pd.read_csv("penguins.csv")
  data, minMaxDF= preprocessing(data)
  trainData, testData = trainTestSplit(data)
  #visualization(data)
  x_train = trainData.drop("species", axis = 1)
  y_train = trainData["species"]
  x_test = testData.drop("species", axis = 1)
  y_test = testData["species"]
  hidden_nodes.append(3)

  # hidden_layers = 1
  #hidden_nodes = [5,3]
  # epochs = 5000
  # eta = 0.001
  # activation_fun = "tanh"
  # bias = 1
  input = np.zeros([5,1])
  print(input)
  if(bias == 1):
    input = np.append(input, 1)
  else:
    input = np.append(input, 0)

  Weights = init_weights(hidden_nodes)
  #trainData, testData = trainTestSplit(data)
  #hidden_layers -> Number of Layer
  Weights = train(Weights,epochs, x_train, y_train, input, eta, activation_fun, bias, hidden_layers)
  print("learned weights")
  print(Weights)
  acc_train(bias, x_train, Weights, activation_fun, hidden_layers, y_train)  
  test(bias, x_test, Weights, activation_fun, hidden_layers, y_test)
  # sampleData = [[13.2, 4500, 2000, "male", 4000]]
  #predictSample(sampleData, x_train, minMaxDF, bias, Weights, activation_fun, hidden_layers)
  return x_train, minMaxDF, bias, Weights, activation_fun, hidden_layers

