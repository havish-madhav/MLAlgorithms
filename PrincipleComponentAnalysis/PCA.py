#importing libraries
impport numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from numpy import linalg
from sklearn import datasets

#loading dataset from library
iris = datasets.load_iris()
a = iris.data[:]
#print(a)

#creating two matricies and inserting values into matrices from the iris data
b=[]
c=[]
for i in range(len(a)):
b.append(a[i][0])
c.append(a[i][1])
mean1=sum(b)/len(b)
for i in range(len(b)):
b[i]=b[i]-mean1
mean2=sum(c)/len(c)
for i in range(len(c)):
c[i]=c[i]-mean2

new_matrix=[]
for j in range(len(b)):
new_matrix.append([b[j],c[j]])

#finding the covarince matrice
cov_mat=numpy.cov(b,c)
print(&quot;covariance matrix is:&quot;,cov_mat)

#calculating the eigenvalues and eigen vectors 
eigen_values=linalg.eig(cov_mat)
print(&quot;eigen values are:&quot;,eigen_values[0])

#forming the new matrice with eigen vales and eigen vectors which represent the spread and dimension of the data
n=numpy.dot(new_matrix,eigen_values[1])

#visualizing the plots before and after the dimension reduction
plt.scatter(b,c)
x=n[:,0]
y=n[:,1]
plt.scatter(x,y)
print(&quot;eigen vectors are:&quot;,eigen_values[1])
