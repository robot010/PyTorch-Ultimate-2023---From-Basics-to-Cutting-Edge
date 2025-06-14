#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import seaborn as sns

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1, 1)
y_list = cars.mpg.values.tolist()
X = torch.from_numpy(X_np)
y = torch.tensor(y_list)

#%% training
w = torch.rand(1, requires_grad=True, dtype=torch.float32)
b = torch.rand(1, requires_grad=True, dtype=torch.float32)

num_epoch=1000
learning_rate=0.001

for epoch in range(num_epoch):
    for i in range(len(X)):
        # forward pass
        y_pred = X[i]*w + b
        # loss
        loss_tensor = torch.pow(y_pred - y[i], 2)
        # backward pass
        loss_tensor.backward()
        # extract losses
        loss_value = loss_tensor.data[0]
        # update w and b
        with torch.no_gard():
            w -= w.grad * learning_rate
            b -= b.grad * learning_rate
            w.grad.zero_() # _ means inplace ops
            b.grad.zero_()
    print(loss_value)

#%% check results
# %%

# %% (Statistical) Linear Regression


# %% create graph visualisation
# make sure GraphViz is installed (https://graphviz.org/download/)
# if not computer restarted, append directly to PATH variable
# import os
# from torchviz import make_dot
# os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin'
# make_dot(loss_tensor)
# %%
