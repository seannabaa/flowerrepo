import pandas as py
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

"""Loading the dataset"""
df = py.read.csv('Iris.csv')
df.head()

#delete a column
df = df.drop(columns = ['id'])
df.head

#to display stats about data
df.describe()

#basic info about datatype
df.info()

#to display number of samples on each class
df['Species'].value_counts()

"""pre-processing the dataset"""
# check for null values
df.isnull().sum()

"""Exploratory Data Analysis"""
#histograms
df['SepalLengthCm'].hist()
df['SepalWidthCm'].hist()
df['PetalLengthCm'].hist()
df['PetalWidthCm'].hist()

#Scatter Plot Characteristics for colours and labels
colors = ['red', 'orange', 'blue']
species = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']

#Scatter Plot for Sepal Length, Width
for i in range (3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label = species[i])
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend()

#Scatter Plot for Petal Length, Width
for i in range (3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label = species[i])
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.legend()

#Scatter Plot for Sepal and Petal Width
for i in range (3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label = species[i])
    plt.xlabel("Sepal Width")
    plt.ylabel("Petal Width")
    plt.legend()

#Scatter Plot for Sepal and Petal Length
for i in range (3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label = species[i])
    plt.xlabel("Sepal Length")
    plt.ylabel("Petal Length")
    plt.legend()


""""Correlation Matrix"""
df.corr()
corr = df.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr, annot=True, ax=ax)

""""Label Encoder"""
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
df.head()

"""Model Training"""
from sklearn.model_selection import train_test_split
#train - 70
#test - 30
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

#logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

#Print metric to get performance
print("Accuracy: ", model.score(x_test, y_test) * 100)

#knn - k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train, y_train)

#print metric to get performance
print("Accuracy: ", model.score(x_test, y_test) * 100)

#decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

#print metric to get performance
print("Accuracy: ", model.score(x_test, y_test) * 100)


