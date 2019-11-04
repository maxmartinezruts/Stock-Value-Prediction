"""
Author: Max Martinez Ruts
Date: October 2019
Idea:

Try to predict the future value of stocks by building a regression model that tries to fit the value over time function
while inducing more error on recent points to underwrite the short-term tendency of the moving average of the market.
The regression model will use as basis functions short samples of previous evolutions of prices found in the market data.
By doing so, maybe some patterns can be built upon other existent patterns encountered in the past.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Import Microsoft stock data
stocks = []
msft = pd.read_csv("MSFT.csv")
stocks.append(msft)

# Retrieve opendings values
ytot = np.array(stocks[0]['Open'])
ltot = len(ytot)

# Range of points to use in the regression model
n_pts = 100


c=0
p=0


bases = []
for i in range(5):
    # Define all base function that are going to be used to build the regression model (get them randomly from the market stock value)
    r = random.randint(0,ltot-n_pts-2)
    bases.append(ytot[r:r+n_pts+1])


"""
Part 1: 
Build a regression model uses as basis functions short samples of previous evolutions of prices found in the market data.
By doing so, maybe some patterns can be built upon other existent patterns encountered in the past.
"""
for k in range(10000):

    # Randomly choose a range of the stock market value to evaluate
    r = random.randint(0,ltot-n_pts-2)

    # Build X by adding the basis functions
    X = [np.ones(n_pts)]
    for base in bases:
        X.append(base[:n_pts])

    # Get y from the market stock value
    y = ytot[r:r+n_pts]
    Y = y

    # Get betas
    X = np.transpose(np.array(X))
    B = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), Y))

    # Determine the prediction of y
    y_pred =  np.zeros(n_pts+1)
    y_pred += np.ones(n_pts+1)*B[0]
    for i in range(len(bases)):
        y_pred += bases[i] * B[i+1]

    # If it correctly predicts the direction of the moving value, add 1 to c. Otherwise substract 1
    if (ytot[r+n_pts]<ytot[r+n_pts-1] and y_pred[n_pts]<y_pred[n_pts-1]) or (ytot[r+n_pts]>ytot[r+n_pts-1] and y_pred[n_pts]>y_pred[n_pts-1]):
        c+=1
    else:
        c-=1
    if (ytot[r+n_pts]<ytot[r+n_pts-1] and y_pred[n_pts]<ytot[r+n_pts-1]) or (ytot[r+n_pts]>ytot[r+n_pts-1] and y_pred[n_pts]>ytot[r+n_pts-1]):
        p+=1
    else:
        p-=1

# Print results of 'success rate' of the model
print(c,p)





"""
Part 2: 
Build a regression model that ponderates higher weights to values closer to the prediction time. By doing so, the short
term behaviour of the market is better represented by the model.
"""

n_vals = 101
y = ytot[:n_vals]
x = np.linspace(0,1,n_vals)
half = np.array(list(np.ones(n_vals)))



balance = [ytot[100]]
times = [1.0]
for k in range(0,100):
    y = ytot[k:k+101]
    Y = y*x*x


    moving_avg = []
    for i in range(3,100):
        moving_avg.append((y[i-3]+y[i-2]+y[i-1]+y[i])/4)

    # Degree of the polynomials basis function
    n_deg = 10

    X = []
    for deg in range(n_deg):
        # By having an exponential x, the values closer to the last values of the range inspected are heavily weighted ocmpared to the values that lie further
        X.append(x*x*x**deg)

    # Get betas
    X = np.transpose(np.array(X))
    B = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),Y))

    # Determine the prediction of y
    y_pred =  np.zeros(121)
    x2 = np.linspace(0,1.2,121)
    for deg in range(n_deg):
        y_pred+= x2**deg * B[deg]

    # Plot values in screen
    plt.plot(x,y)
    plt.plot(x2,y_pred)
    plt.plot(x2[0:120],ytot[0+k:120+k], 'ro')
    plt.plot(1.01,y_pred[101],'go')
    plt.plot(1.00,y_pred[100],'go')
    plt.plot(1.00, ytot[100+k], 'yo')
    plt.plot(1.01, ytot[100+k+1], 'yo')


    # Simply sell if prediction of tomorrow is lower than todays opening, and viceversa
    if y_pred[101] < y_pred[100]:
        # Sell
        balance.append(balance[-1])
    else:
        # Buy
        balance.append(balance[-1]+ytot[100+k+1]-ytot[100+k])

    times.append(1.01+k/100)
    plt.plot(np.array(times)-k/100,balance)
    plt.grid()
    plt.show()
