import pandas as pd
import numpy as np
import matplotlib as plt
data=pd.read_csv("Credit_Card_Applications.csv")
x1=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
ms=MinMaxScaler()
x=ms.fit_transform(x1)

from minisom import MiniSom
SOM=MiniSom(x=10,y=10,input_len=15)
SOM.random_weights_init(x)
SOM.train_random(data=x,num_iteration=100)
"""##Visualizing the results"""

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(SOM.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x2 in enumerate(x):
    w = SOM.winner(x2)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

"""## Finding the frauds"""

mappings = SOM.win_map(x)
print(mappings[(7,8)])
# frauds = np.concatenate((mappings[(4,7)], mappings[(7,8)]), axis = 0)
# frauds = ms.inverse_transform(frauds)
