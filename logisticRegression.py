import numpy as np
import pandas as pd

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

learning_rate = 0.0001
no_of_iterations =10
k = 0.0001
x = pd.read_csv('Datasets/data_logistic.csv')

print("Running On Dataset-" + "data_logistic.csv")
new = []
for i in zip(x['variance'], x['skewness'], x['curtosis'], x['entropy'], x['c']):
    new.append((i[0], i[1], i[2], i[3], i[4]))
    
new = np.array(new)

data_size = len(new)

data = []
for i in zip(x['variance'], x['skewness'], x['curtosis'], x['entropy'], x['c']):
    data.append((i[0], i[1], i[2], i[3], 1))
    
data = np.array(data)

#sepearating the third column into two classes
class0 = np.array([i for i in new if i[4]==0])
class1 = np.array([i for i in new if i[4]==1])

w = np.ndarray((5, 1))
for i in range(0, 5):
	w[i]=0
#Assigning the sign and calculating the error
for iter in range(0, no_of_iterations):
    for row in range(len(data)):
        x = np.ndarray((5, 1), buffer = np.array([data[row][0], data[row][1], data[row][2], data[row][3], data[row][4]]))   
        ans = sigmoid(np.matmul(np.transpose(w), x) )
        if( (ans >=0.5 and new[row][4] ==0) or (ans <0.5 and new[row][4] ==1)):
        	w = w + (learning_rate)*((1/data_size)*(new[row][4]-ans)*x 	+k*w)

    

print(w)

print("Testing on Logistic Test data....")


y = pd.read_csv('Datasets/data_logistic.csv')

new = []
for i in zip(y['variance'], y['skewness'], y['curtosis'], y['entropy'], y['c']):
    new.append((i[0], i[1], i[2], i[3], i[4]))

new = np.array(new)

data_size = len(new)

data = []
for i in zip(y['variance'], y['skewness'], y['curtosis'], y['entropy'], y['c']):
    data.append((i[0], i[1], i[2], i[3], 1))
    
data = np.array(data)
#print(data)
cnt1=0
cnt2=0
cnt3=0
cnt4=0
for row in range(len(data)):
    x = np.ndarray((5, 1), buffer = np.array([data[row][0], data[row][1], data[row][2], data[row][3], data[row][4]]))
    ans = sigmoid(np.transpose(w).dot(x))
    if(ans >= 0.5 and new[row][4]==1):
    	cnt1 = cnt1+1;
    if(ans >= 0.5 and new[row][4]==0):
    	cnt2 = cnt2+1;
    if(ans < 0.5 and new[row][4]==1):
    	cnt3 = cnt3+1;
    if(ans < 0.5 and new[row][4]==0):
    	cnt4 = cnt4+1;

print(cnt1)
print(cnt2)
print(cnt3)
print(cnt4)

print("Acuracy is - " + str((cnt1+cnt4)/(cnt1+cnt2+cnt4+cnt3) ) )
    
