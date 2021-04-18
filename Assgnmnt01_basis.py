#License: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron(object):
    """Perceptron

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight initialization.

    Attributes
    -----------
    w : 1d-array
      Weights after fitting.
    errors : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors = []

        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w[1:] += update * xi
                self.w[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# ### Reading-in the Iris data
df = pd.read_csv('https://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data', header=None)

def featureSelector(combination):
      """Helper Function which enables to select any 1 of the 4 possible features combinations which are:

        Combination 1 - Features 1, 2 and 3
        Combination 2 - Features 1, 3 and 4
        Combination 3 - Features 2, 3 and 4
        Combination 4 - Features 1, 2 and 4

        where;
        feature  1 = "Sepal Length"
        feature  2 = "Sepal Width"
        feature  3 = "Petal Length"
        feature  4 = "Petal Width"      
      """      
      if combination == 1:
            X = df.iloc[0:100, [0, 1, 2]].values
      elif combination == 2:
            X = df.iloc[0:100, [0, 2, 3]].values
      elif combination == 3:
            X = df.iloc[0:100, [1, 2, 3]].values
      elif combination == 4:
            X = df.iloc[0:100, [0, 1, 3]].values
      else:
            print("Incorrect Number Entered")
            return None
      return X


ppn = Perceptron(eta=0.1, n_iter=4)

# select setosa and versicolor and assign -1 as class label if Setosa and 1 otherwise
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)


# Plots of all the different combinations' errors
fig, axs = plt.subplots(2, 2)

# Combination 1 (Features 1-2-3) plot 
x = featureSelector(1)
ppn.fit(x, y)
print("Combination 1 (features 1-2-3) errors :", ppn.errors)
axs[0, 0].plot(range(1, len(ppn.errors) + 1), ppn.errors,marker='o')
axs[0,0].set_ylim([-0.1, max(ppn.errors) + 0.1])
axs[0, 0].set_title('Features 1-2-3')

# Combination 2 (Features 1-3-4) plot 
x = featureSelector(2)
ppn.fit(x, y)
print("Combination 2 (features 1-3-4) errors :", ppn.errors)
axs[0, 1].plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o', color = 'r')
axs[0,1].set_ylim([-0.1, max(ppn.errors) + 0.1])
axs[0, 1].set_title('Features 1-3-4')

# Combination 3 (Features 2-3-4) plot 
x = featureSelector(3)
ppn.fit(x, y)
print("Combination 3 (features 2-3-4) errors :", ppn.errors)
axs[1, 0].plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o' , color = 'g')
axs[1,0].set_ylim([-0.1, max(ppn.errors) + 0.1])
axs[1, 0].set_title('Features 2-3-4')

# Combination 4 (Features 1-2-4) plot 
x = featureSelector(4)
ppn.fit(x, y)
print("Combination 4 (features 1-2-4) errors :", ppn.errors)
axs[1, 1].plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o', color = 'm' )
axs[1,1].set_ylim([-0.1, max(ppn.errors) + 0.1])
axs[1, 1].set_title('Features 1-2-4')

for ax in axs.flat:
    ax.set(xlabel='Epochs (Iterations)', ylabel='Number of Errors')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()