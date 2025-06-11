import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from dtwalign import dtw

plt.rcParams['pdf.fonttype'] = 42

vel=np.load('./result/LSTM/ESA/ESA/0614/prediction.npy')
a=scipy.io.loadmat('./dataset/ESA/0614.mat')
label=a['trial_velocity']

vel=vel[:,np.where(a['trial_mask']==73)[1]]
label=label[:,np.where(a['trial_mask']==73)[1]]

pos=np.cumsum(vel,axis=1)
posl=np.cumsum(label,axis=1)

res=dtw(vel.T,label.T)
x_warping_path = res.get_warping_path(target="query")
plt.plot(vel.T[x_warping_path,0],label='aligned prediction to label')
plt.plot(vel.T[:,0],label='prediction')
plt.plot(label.T[:,0],label='label')
plt.title('X-axis')
plt.legend()
plt.savefig('C:/Users/24233/Desktop/result/3p4-1.pdf',dpi=300)
plt.show()
plt.close()
plt.plot(vel.T[x_warping_path,1],label='aligned prediction to label')
plt.plot(vel.T[:,1],label='prediction')
plt.plot(label.T[:,1],label='label')
plt.title('Y-axis')
plt.legend()
plt.savefig('C:/Users/24233/Desktop/result/3p4-2.pdf',dpi=300)
plt.show()
plt.close()

posw=np.cumsum(vel[:,x_warping_path],axis=1)

plt.plot(pos[0],pos[1])
plt.axis('off')
ax = plt.gca()
ax.set_aspect(1)
plt.savefig('C:/Users/24233/Desktop/result/3p4-3.pdf',dpi=300)
plt.show()
plt.close()
plt.plot(posw[0],posw[1])
plt.axis('off')
ax = plt.gca()
ax.set_aspect(1)
plt.savefig('C:/Users/24233/Desktop/result/3p4-4.pdf',dpi=300)
plt.show()
plt.close()
plt.plot(posl[0],posl[1])
plt.axis('off')
ax = plt.gca()
ax.set_aspect(1)
plt.savefig('C:/Users/24233/Desktop/result/3p4-5.pdf',dpi=300)
plt.show()
plt.close()