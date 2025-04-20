import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pprint import pprint

# Read the data and prepare for training 
df = pd.read_csv('logistic-regression-purchase-prediction_no-noise.csv')

# --- Apply feature scaling
X = np.array(df.iloc[:, 0:2])                      # Features: Age, Income
Y = np.array(df['Purchased']).reshape(-1, 1)       # Labels: Purchased (0 or 1)

X_n = (X - X.mean(axis=0)) / X.std(axis=0)         # Feature scaling (Z-score)

# --- Hyperparameters
epochs = 1000
m = X_n.shape[0]    # Number of samples
alpha = 0.001       # Learning rate

# --- Parameters initialization
b = 0
w = np.zeros((X_n.shape[1], 1))  # Shape: (2, 1) for 2 features

# --- Cost function for logistic regression
def cost_value(m, Y, y_pred):
    cost = -(1/m) * np.sum(Y * np.log(y_pred + 1e-15) + (1 - Y) * np.log(1 - y_pred + 1e-15))
    return cost

# --- Training loop (gradient descent)
cost_history = []
for i in range(epochs):
    z = np.dot(X_n, w) + b
    y_pred = 1 / (1 + (np.e)**-z)

    cost = cost_value(m, Y, y_pred)
    cost_history.append(cost)

    db = (1/m) * np.sum(y_pred - Y)
    dw = (1/m) * np.dot(X_n.T, (y_pred - Y))

    b = b - alpha * db
    w = w - alpha * dw

# --- plot the data 
plt.figure()
plt.scatter(X_n[Y[:,0]==1][:,0],X_n[Y[:,0]==1][:,1],marker='+',color='r',label='Purchased Ones')
plt.scatter(X_n[Y[:,0]==0][:,0],X_n[Y[:,0]==0][:,1],marker='o',color='b',label='Un Purchased Ones')
plt.title("The Data With Scalling Features")
plt.xlabel('Age')
plt.ylabel('Income')

# --- plot the decesion boundry
# ---z=w1x1+w2x2+b so at z=0 get  0=w1x1+w2x2+b then x1=-(b+w2x2)/w1
x2=np.linspace(np.min(X_n[:,1]),np.max(X_n[:,1]),100)
x1 = (-(b+w[1]*x2))/(w[0])
plt.plot(x1,x2,color='k',label='decision boundry')
plt.legend()
plt.show()

# --- calculate the module accuracy !
def Module_accuracy(y_pred,Y):
    bool_ = (y_pred >=.5).astype(int)
    accuracy = np.mean(bool_ == Y)
    return accuracy*100

print(f'{Module_accuracy(y_pred,Y)}%')