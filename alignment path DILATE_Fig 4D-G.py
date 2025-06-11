import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy
import scipy.io
import torch
from dilate_loss import*
from dtw import*

plt.rcParams['pdf.fonttype'] = 42

def time_interpolation(x,n):
    m=x.shape[1]
    b=np.zeros((x.shape[0],m+1))
    b[:,1:]=np.cumsum(x,axis=1)
    out=np.zeros((x.shape[0],n+1))
    out[0]=np.interp(np.arange(n+1)*m/n,np.arange(m+1),b[0])
    out[1]=np.interp(np.arange(n+1)*m/n,np.arange(m+1),b[1])
    return out[:,1:]-out[:,:-1]

def get_path(e):
    n=e.shape[1]
    m=np.zeros((n+2,n+2))
    m[1:-1,1:-1]=e
    path=[[1,1]]
    i=1
    j=1
    while True:
        q=np.array([m[i+1,j],m[i+1,j+1],m[i,j+1]]).max()
        if q==m[i+1,j]:
            i=i+1
            path.append([i,j])
        elif q==m[i+1,j+1]:
            i=i+1
            j=j+1
            path.append([i,j])
        else:
            j=j+1
            path.append([i,j])
        if i==n and j==n:
            break
    return np.array(path,dtype=np.int16).T

daylist=['0614','0616','0623','0624','0630','0701']
kt=[    0, 10920, 23940, 36540, 48780, 63180, 74100]
length=140
vel_l=[]
vel_p=[]
for i,day in enumerate(daylist):
    dataset=scipy.io.loadmat('./dataset/ESA/{0}.mat'.format(day))
    vel=np.load('./result/LSTM/ESA/ESA_norm_DILATE/alpha_0.5_gamma_0.001/{0}/prediction.npy'.format(day),allow_pickle=True)
    for j in range(90):
        t=np.where(dataset['trial_mask'][0]==(j+1))[0]
        vel_l.append(torch.tensor(time_interpolation(dataset['trial_velocity'][:,t],length),dtype=torch.float32))
        vel_p.append(torch.tensor(time_interpolation(vel[:,t],length),dtype=torch.float32))

matrix_list=[]
path_list=[]

for i in range(len(vel_l)):
    pm=get_path_matrix(vel_p[i],vel_l[i])
    matrix_list.append(pm.clone())
    path_list.append(get_path(pm))

for i in range(len(path_list)):
    plt.plot(path_list[i][0],path_list[i][1],linewidth=0.5)
plt.axhline(20,c='black',linewidth=0.5, linestyle='--')
plt.axhline(90,c='black',linewidth=0.5, linestyle='--')
plt.gca().invert_yaxis()
plt.savefig('C:/Users/24233/Desktop/result/D-2-1-1.pdf', dpi = 300)
plt.show()
plt.close()

sum_matrix=torch.zeros((length,length))
for i in matrix_list:
    sum_matrix+=i
sum_matrix=sum_matrix/len(matrix_list)
plt.plot(sum_matrix[20],label='t=20')
plt.plot(sum_matrix[90],label='t=90')
plt.legend()
plt.savefig('C:/Users/24233/Desktop/result/D-2-1-2.pdf', dpi = 300)
plt.show()
plt.close()

sum_path=np.concatenate(path_list, axis=1)
bias=sum_path[0]-sum_path[1]
plt.hist(bias,bins=np.arange(bias.min()-0.5,bias.max()+0.5,1),density=True)
plt.axvline(bias.mean(),c='black',linewidth=0.5, linestyle='--')
plt.xlim(-40,40)
plt.savefig('C:/Users/24233/Desktop/result/D-2-1-3.pdf', dpi = 300)
plt.show()
plt.close()

std=[]
for i in range(len(path_list)):
    b=path_list[i][0]-path_list[i][1]
    std.append(b.var()**0.5)
std=np.array(std)
plt.hist(std,bins=np.arange(0,int(std.max())+1,1),density=True)
plt.axvline(std.mean(),c='black',linewidth=0.5, linestyle='--')
plt.xlim(0,12)
plt.savefig('C:/Users/24233/Desktop/result/D-2-1-4.pdf', dpi = 300)
plt.show()
plt.close()