# "- create a new github repo called Regularization-Experiments

# you should make at least 3 git commits during this challenge

# create a (20 by 1) matrix of random x values between 0 and 1
# these will be our design matrix (20 examples, 1 feature)


import numpy as np 
import plotly.express as px
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

mat = np.random.rand(20,1)
# print (mat)


# defing some function which takes in those single feature examples and returns a new design matrix with an extra column
#  representing the x-squared values
# generalise this function to be able to return you features which are powers of x up to some number

def new_matrix (features, xpower):
    new_array = []
    for feature in features:
        # print ("FEATURE IS")
        # print (feature)
        xsq =  feature**xpower
        row = [feature, xsq]
        #print ("ROW IS ")
        #print (row)
        new_array.append (row)
    return np.array (new_array)


# print (new_matrix(mat, 3))


# define a function which computes a label such as y = 2 + x + 0.2*x^2 + 0.1*x^2
# visualise this on a X-Y graph and play around with the coefficients until you get a function that is not too boring (linear)
def output1 (data):
    y_points = []
    for x in data:
        y = 1 + x + 20*x**2 
        y_points.append (y)
    return np.array(y_points)
# graph = px.scatter(mat, output(mat))
# graph.show()
# split the data into train, val and test sets


def output2 (data):
    y_points = []
    for x in data:
        y = 1 + x + 20*x**2 + x ** 3 + 10 * x ** 4
        y_points.append (y)
    return np.array(y_points)


def output3 (data):
    y_points = []
    for x in data:
        y = 1 + x + 20*x**2 + x ** 3 + 3* x ** 4 + 5 * x ** 7 + 8 *x**8
        y_points.append (y)
    return np.array(y_points)

def output4 (data):
    y_points = []
    for x in data:
        y = 1 + x + 20*x**2 + x ** 3 + 5* x ** 4 + 2 * x ** 7 + 11 *x**8 + 12 * x**10 + 3 *x**11
        y_points.append (y)
    return np.array(y_points)


X_train, X_test, y_train, y_test = model_selection.train_test_split(mat, output4(mat), test_size=0.3)
X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5)
m1 = LinearRegression().fit (X_train,y_train)



# Make predictions using the testing set
m1_pred = m1.predict(X_train)

print (m1_pred)

# fit a model to these labels, firstly just passing your model the original features (x^1)
# visualise the predictions against the label you should see that the model is underfit

import matplotlib.pyplot as plt

# Plot outputs
plt.scatter(X_train, y_train, color="black")
plt.plot(X_train, m1_pred, color="blue", linewidth=1)
plt.xticks(())
plt.yticks(())
plt.show()

# now train a series of models on the design matrix that contain sequentially increasing powers of x
# include powers of x way above those which your labels are based on
# e.g. go up to features where x^12 is included
# the models trained on these should overfit the data (easy to do if you make the train set small)

print ("Done, see above outputs 1-4")


# grid search over the capacity hyperparam (which power of x is included) to evaluate each model on the train and val set
# dicsuss: what were the results?"