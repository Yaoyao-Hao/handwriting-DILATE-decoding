import numpy as np
import torch
import torch.nn as nn
from dilate_loss import*
import matplotlib.pyplot as plt

a=np.cos(np.linspace(-np.pi, np.pi,21))+1
label=np.zeros(100)
label[40:61]+=a
true=np.zeros(100)
true[30:51]+=a

labelt=torch.tensor(label).view(1,-1)
truet=torch.tensor(true).view(1,-1)
plt.plot(labelt[0],label="'Actual'")
plt.plot(truet[0],label='Prompted')
plt.legend()
plt.show()
plt.close()

mseab=torch.tensor([1.,0.])
dilateab=mseab.clone()
mseab=nn.Parameter(mseab)
mseopt=torch.optim.SGD([mseab],lr=0.001)
mse=nn.MSELoss()
dilateab=nn.Parameter(dilateab)
dilateopt=torch.optim.SGD([dilateab],lr=0.001)

lossmse=np.zeros((101,61))
lossdilate=np.zeros((101,61))
for i in range(101):
    for j in range(61):
        a=(i-50)/10
        b=(j-30)/10
        lossmse[i,j]=float(mse(a*truet+b,labelt))
        lossdilate[i,j]=float(dilate_loss(a*truet+b,labelt))
        
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(np.linspace(-3,3,61),np.linspace(-5,5,101))
ax.plot_surface(X,Y, lossmse, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_title('MSE')
plt.show()
plt.close()

ax = plt.axes(projection='3d')
X, Y = np.meshgrid(np.linspace(-3,3,61),np.linspace(-5,5,101))
ax.plot_surface(X,Y, lossdilate, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_title('DILATE')
plt.show()
plt.close()