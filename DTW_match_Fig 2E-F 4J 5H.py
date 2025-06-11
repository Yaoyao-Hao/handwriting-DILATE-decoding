import os
import numpy as np
import scipy.io
import time
from dtw import*

def cut(feature_list,mask):
    '''
    Based on mask segmentation features (temporal sequence)
    '''
    if type(feature_list)==list and len(feature_list)!=mask.shape[0]:
        start_mark=0
        end_mark=start_mark
        n=len(mask)
        m=len(feature_list)
        res=[[] for i in range(m)]
        while True:
            start_mark=end_mark
            while True:
                end_mark=end_mark+1
                if end_mark==n or mask[start_mark]!=mask[end_mark]:
                    break
            for i in range(m):
                res[i].append(feature_list[i][:,start_mark:end_mark])
            if end_mark==n:
                break
    else:
        start_mark=0
        end_mark=start_mark
        res=[]
        while True:
            start_mark=end_mark
            while True:
                end_mark=end_mark+1
                if end_mark==len(mask) or mask[start_mark]!=mask[end_mark]:
                    break
            res.append(feature_list[:,start_mark:end_mark])
            if end_mark==len(mask):
                break
    return res

def time_interpolation(x,n):
    m=x.shape[1]
    b=np.zeros((x.shape[0],m+1))
    b[:,1:]=np.cumsum(x,axis=1)
    out=np.zeros((x.shape[0],n+1))
    out[0]=np.interp(np.arange(n+1)*m/n,np.arange(m+1),b[0])
    out[1]=np.interp(np.arange(n+1)*m/n,np.arange(m+1),b[1])
    return out[:,1:]-out[:,:-1]

def dtw_d(fit,true):
    return dtw(fit,true,distance_only=True).distance
    
if __name__ == "__main__":
    length=300
    daylength=[0,10920, 23940, 36540, 48780, 63180, 74100]
    pt='ESA'
    pt1='ESA/norm_ESA_combine'
    daylist=['0614','0616','0623','0624','0630','0701']
    t=0
    alldata={}
    for w in range(len(daylist)):
        day=daylist[w]
        a=scipy.io.loadmat('./dataset/'+pt+'/'+day+'.mat')
        # prediction=np.load('./result/LSTM/'+pt1+'/prediction.npy',allow_pickle=True)[:,daylength[w]:daylength[w+1]]
        prediction=np.load('./result/LSTM/'+pt1+'/'+day+'/prediction.npy',allow_pickle=True)
        trial_velocity,prediction=cut([a['trial_velocity'],prediction],a['trial_mask'][0])
        k=[]
        for i in range(a['target_hanzi'].shape[1]):
            zi=scipy.io.loadmat('./character_set/dataset/180/{0}.mat'.format(a['target_hanzi'][0,i][0]))
            p=np.cumsum(zi['trial_velocity'],axis=1)
            zi['trial_velocity'][0]=zi['trial_velocity'][0]/(zi['trial_velocity'][0].max()-zi['trial_velocity'][0].min())
            zi['trial_velocity'][1]=zi['trial_velocity'][1]/(zi['trial_velocity'][1].max()-zi['trial_velocity'][1].min())
            if 'pos' in pt1:
                k.append(np.cumsum(zi['trial_velocity'],axis=1))
            else:
                k.append(zi['trial_velocity'])
        
        try:
            alldata['pre'].extend(prediction)
            alldata['tri']=np.concatenate([alldata['tri'],(a['trial_target'].astype(np.uint32)+30*t)],axis=0)
            alldata['vel'].extend(k)
            t=t+1
        except:
            alldata['pre']=prediction
            alldata['vel']=k
            alldata['tri']=a['trial_target'].astype(np.uint32)
            t=t+1
    
    path=os.listdir("./character_set/dataset/820/")
    for i in path:
        alldata['vel'].append(scipy.io.loadmat('./character_set/dataset/820/'+i)['trial_velocity'])
    
    timeres=np.zeros((540,1))
    res_matrix=np.zeros((len(alldata['pre']),len(alldata['vel'])))
    res=np.zeros((5,len(alldata['pre'])))
    for i in range(len(alldata['pre'])):
        start_time=time.time_ns()
        mse=np.zeros(len(alldata['vel']))
        
        for j in range(len(alldata['vel'])):
            
            # DTW distance
            inp=time_interpolation(alldata['pre'][i],length)
            inp=inp.T
            inp=(inp-inp.mean(0))/inp.var(0)**0.5
            rea=time_interpolation(alldata['vel'][j],length)
            rea=rea.T
            rea=(rea-rea.mean(0))/rea.var(0)**0.5
            mse[j]=-dtw_d(inp,rea)
            
            # interpolation corrcoef
            # inp=time_interpolation(alldata['pre'][i],length)
            # rea=time_interpolation(alldata['vel'][j],length)
            # mse[j]=(np.corrcoef(inp[0],rea[0])[0,1]+np.corrcoef(inp[1],rea[1])[0,1])/2
            
        sor=np.argsort(mse)
        res_matrix[i]=mse
        for m in range(5):
            if (alldata['tri'][i,0]-1) in sor[-m-1:]:
                res[m,i]=1
        end_time=time.time_ns()
        timeres[i,0]=end_time=start_time
    acc=(res.sum(1))/(res.shape[1])
    
    res=np.zeros((9,1000))
    for k,n in enumerate([180,300,400,500,600,700,800,900,1000]):
        for j in range(1000):#Randomly select 1000 characters
            t=np.arange(180,1000)
            td=np.random.choice(t,1000-n,replace=False)
            resample=np.delete(res_matrix, td,axis=1)
            for i in range(540):
                sor=np.argsort(resample[i])
                if (alldata['tri'][i,0]-1) in sor[-1:]:
                    res[k,j]+=1
    res=res/540