import numpy as np
import math
import scipy.signal as signal
import scipy.io


def wave_filter(sequence,cof,sf,mod,N=4):
    if len(sequence.shape)==1:
        sequence=sequence.reshape(1,-1)
    b, a = signal.butter(N, cof, btype=mod,fs=sf)
    filtedData = signal.filtfilt(b, a, sequence)
    return filtedData



def get_esa(raw_data_path,dataset_path):
    a=300
    b=6000
    c=12
    windowlength=200
    dataset=scipy.io.loadmat(dataset_path)
    timestamp=np.load(raw_data_path+'timestamp.npy',allow_pickle=True)
    median=np.load(raw_data_path+'median.npy',allow_pickle=True)[0]
    start=np.load(raw_data_path+'start.npy',allow_pickle=True)
    end=np.load(raw_data_path+'end.npy',allow_pickle=True)
    late=np.load(raw_data_path+'late.npy',allow_pickle=True)
    
    middle_fre=1000
    length=[((dataset['trial_mask']==i).sum()//20)*middle_fre for i in range(1,91)]
    down=30000//middle_fre
    
    esa=np.zeros_like(dataset['bined_spk'],dtype=np.float64)

    for i in range(96):
        data=np.load(raw_data_path+'{0}.npy'.format(i),allow_pickle=True)
        
        #Pretreatment
        data=data-data.mean()
        data=data-median
        
        #Extract ESA
        data=wave_filter(data,a,30000,'highpass',1)[0]
        data=np.abs(data)
        data=wave_filter(data,c,30000,'lowpass',1)[0]
        
        trial_data=[]
        for j in range(start.shape[0]):
            fragment=data[start[j]+late*30-windowlength*15-timestamp:start[j]+late*30+length[j]*down+windowlength*15-timestamp:down]
            res=np.zeros((length[j]//middle_fre)*20)
            
            #Sliding average window
            for k in range((length[j]//middle_fre)*20):
                res[k]=(fragment[k*(middle_fre//20):k*(middle_fre//20)+middle_fre*windowlength//1000]).mean()
            trial_data.append(res)
        esa[i]=np.concatenate(trial_data)
    dataset['bined_spk']=esa
    return dataset

def get_sbp(raw_data_path,dataset_path):
    a=300
    b=1000
    c=12
    windowlength=200
    dataset=scipy.io.loadmat(dataset_path)
    timestamp=np.load(raw_data_path+'timestamp.npy',allow_pickle=True)
    median=np.load(raw_data_path+'median.npy',allow_pickle=True)[0]
    start=np.load(raw_data_path+'start.npy',allow_pickle=True)
    end=np.load(raw_data_path+'end.npy',allow_pickle=True)
    late=np.load(raw_data_path+'late.npy',allow_pickle=True)
    
    middle_fre=2000
    length=[((dataset['trial_mask']==i).sum()//20)*middle_fre for i in range(1,91)]
    down=30000//middle_fre
    
    sbp=np.zeros_like(dataset['bined_spk'],dtype=np.float64)

    for i in range(96):
        data=np.load(raw_data_path+'{0}.npy'.format(i),allow_pickle=True)
        
        #Pretreatment
        data=data-data.mean()
        data=data-median
        
        #Extract SBP
        data=wave_filter(data,[a,b],30000,'highpass',2)[0]
        data=np.abs(data)
        
        trial_data=[]
        for j in range(start.shape[0]):
            fragment=data[start[j]+late*30-windowlength*15-timestamp:start[j]+late*30+length[j]*down+windowlength*15-timestamp:down]
            res=np.zeros((length[j]//middle_fre)*20)
            
            #Sliding average window
            for k in range((length[j]//middle_fre)*20):
                res[k]=(fragment[k*(middle_fre//20):k*(middle_fre//20)+middle_fre*windowlength//1000]).mean()
            trial_data.append(res)
        sbp[i]=np.concatenate(trial_data)
    dataset['bined_spk']=sbp
    return dataset

def get_cmua(raw_data_path,dataset_path):
    a=300
    b=6000
    c=100
    windowlength=200
    dataset=scipy.io.loadmat(dataset_path)
    timestamp=np.load(raw_data_path+'timestamp.npy',allow_pickle=True)
    median=np.load(raw_data_path+'median.npy',allow_pickle=True)[0]
    start=np.load(raw_data_path+'start.npy',allow_pickle=True)
    end=np.load(raw_data_path+'end.npy',allow_pickle=True)
    late=np.load(raw_data_path+'late.npy',allow_pickle=True)
    
    middle_fre=1000
    length=[((dataset['trial_mask']==i).sum()//20)*middle_fre for i in range(1,91)]
    down=30000//middle_fre
    
    cmua=np.zeros_like(dataset['bined_spk'],dtype=np.float64)

    for i in range(96):
        data=np.load(raw_data_path+'{0}.npy'.format(i),allow_pickle=True)
        
        #Pretreatment
        data=data-data.mean()
        data=data-median
        
        #Extract SBP
        data=wave_filter(data,[a,b],30000,'highpass',3)[0]
        data=data**2
        data=wave_filter(data,c,30000,'lowpass',3)[0]
        data[data<0.]=0.
        data=data**0.5
        
        trial_data=[]
        for j in range(start.shape[0]):
            fragment=data[start[j]+late*30-windowlength*15-timestamp:start[j]+late*30+length[j]*down+windowlength*15-timestamp:down]
            res=np.zeros((length[j]//middle_fre)*20)
            
            #Sliding average window
            for k in range((length[j]//middle_fre)*20):
                res[k]=(fragment[k*(middle_fre//20):k*(middle_fre//20)+middle_fre*windowlength//1000]).mean()
            trial_data.append(res)
        cmua[i]=np.concatenate(trial_data)
    dataset['bined_spk']=cmua
    return dataset

def get_lmp(raw_data_path,sua_path,highpass_f):
    windowlength=200
    dataset=scipy.io.loadmat(sua_path)
    timestamp=np.load(raw_data_path+'timestamp.npy',allow_pickle=True)
    median=np.load(raw_data_path+'median.npy',allow_pickle=True)[0]
    start=np.load(raw_data_path+'start.npy',allow_pickle=True)
    end=np.load(raw_data_path+'end.npy',allow_pickle=True)
    late=np.load(raw_data_path+'late.npy',allow_pickle=True)
    
    middle_fre=2000
    length=[((dataset['trial_mask']==i).sum()//20)*middle_fre for i in range(1,91)]
    down=30000//middle_fre
    
    lmp=np.zeros_like(dataset['bined_spk'],dtype=np.float64)
    for i in range(96):
        data=np.load(raw_data_path+'{0}.npy'.format(i),allow_pickle=True)
        
        #Pretreatment
        data=data-data.mean()
        data=data-median
        
        #Extract LMP
        data=wave_filter(data,300,30000,'lowpass',4)[0]
        data=data[::down]
        data[data>300.]=300.
        data[data<-300.]=-300.
        
        
        trial_data=[]
        for j in range(start.shape[0]):
            fragment=data[start[j]+late*30-windowlength*15-timestamp:start[j]+late*30+length[j]*down+windowlength*15-timestamp:down]
            res=np.zeros((length[j]//middle_fre)*20)
            
            #Sliding average window
            for k in range((length[j]//middle_fre)*20):
                res[k]=(fragment[k*(middle_fre//20):k*(middle_fre//20)+(middle_fre*windowlength)//1000]**2).mean()
            trial_data.append(res)
        lmp[i]=np.concatenate(trial_data)
    dataset['bined_spk']=lmp
    return dataset

def get_power_lfp(raw_data_path,sua_path,bandpass_f):
    if bandpass_f[0]<20:
        windowlength=math.ceil(1000/bandpass_f[0])
    else:
        windowlength=50
    dataset=scipy.io.loadmat(sua_path)
    timestamp=np.load(raw_data_path+'timestamp.npy',allow_pickle=True)
    median=np.load(raw_data_path+'median.npy',allow_pickle=True)[0]
    start=np.load(raw_data_path+'start.npy',allow_pickle=True)
    end=np.load(raw_data_path+'end.npy',allow_pickle=True)
    late=np.load(raw_data_path+'late.npy',allow_pickle=True)

    middle_fre=2000
    length=[((dataset['trial_mask']==i).sum()//20)*middle_fre for i in range(1,91)]
    down=30000//middle_fre

    lfp=np.zeros_like(dataset['bined_spk'],dtype=np.float64)
    mul=4

    for i in range(96):
        data=np.load(raw_data_path+'{0}.npy'.format(i),allow_pickle=True)

        #Pretreatment
        data=data-data.mean()
        data=data-median

        #Extract power LFP
        data=wave_filter(data,500,30000,'lowpass',4)[0]
        sd=(data.var())**0.5
        mean=data.mean()
        data[data>mean+mul*sd]=mean+mul*sd
        data[data<mean-mul*sd]=mean-mul*sd
        data=wave_filter(data,bandpass_f,30000//down,'bandpass',3)

        trial_data=[]
        for j in range(start.shape[0]):
            mres=data[start[j]+late*30-windowlength*15-timestamp:start[j]+late*30+length[j]*down+windowlength*15-timestamp:down]
            res=np.zeros((length[j]//middle_fre)*20)

            #Sliding average window
            for k in range((length[j]//middle_fre)*20):
                res[k]=(mres[k*(middle_fre//20):k*(middle_fre//20)+middle_fre*windowlength//1000]**2).mean()
            trial_data.append(res)
        lfp[i]=np.concatenate(trial_data)
    dataset['bined_spk']=lfp
    return dataset

def get_mua(raw_data_path,sua_path,threshold):
    windowlength=200
    threshold = -4.5
    dataset=scipy.io.loadmat(sua_path)
    timestamp=np.load(raw_data_path+'timestamp.npy',allow_pickle=True)
    median=np.load(raw_data_path+'median.npy',allow_pickle=True)[0]
    start=np.load(raw_data_path+'start.npy',allow_pickle=True)
    end=np.load(raw_data_path+'end.npy',allow_pickle=True)
    late=np.load(raw_data_path+'late.npy',allow_pickle=True)
    
    middle_fre=30000
    length=[((dataset['trial_mask']==i).sum()//20)*middle_fre for i in range(1,91)]
    down=30000//middle_fre
    
    mua=np.zeros_like(dataset['bined_spk'],dtype=np.float64)
    print('dataset{0}'.format(sua_path[20:24]))
    for i in range(96):
        data=np.load(raw_data_path+'{0}.npy'.format(i),allow_pickle=True)
        
        #Pretreatment
        data=data-data.mean()
        data=data-median
        
        #Extract power LFP
        data=wave_filter(data,[250,5000],30000,'bandpass',3)[0]
        rms = np.sqrt(np.mean(np.square(data)))
        tmp = (data>(threshold*rms)).astype(np.int64)
        data=(np.diff(tmp)==1)
        
        trial_data=[]
        for j in range(start.shape[0]):
            mres=data[start[j]+late*30-timestamp-windowlength*15:start[j]+late*30+length[j]*down-timestamp+windowlength*15:down]
            res=np.zeros((length[j]//middle_fre)*20,dtype=np.int64)
            
            #Sliding average window
            for k in range((length[j]//middle_fre)*20):
                res[k]=mres[k*(middle_fre//20)*down:(k*(middle_fre//20)+middle_fre*windowlength//1000)*down].sum()
            trial_data.append(res)
        mua[i]=np.concatenate(trial_data)
    dataset['bined_spk']=mua
    return dataset

