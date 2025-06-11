import os
import time
import scipy.io
import numpy as np
import scipy
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class LSTM_Net(nn.Module):

    def __init__(self,input_size,hidden_size,output_size,bidirectional):
        super(LSTM_Net,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.lstm=nn.LSTM(input_size,hidden_size,1,batch_first=True,bidirectional=bidirectional)
        self.fc1=nn.Linear(hidden_size, output_size)

    def forward(self,x,hc=0):
        res=x.t()
        res,hcn=self.lstm(res.unsqueeze(0),hc)
        res=self.fc1(F.elu(res))
        hcn=tuple([i.detach() for i in hcn])
        return res.squeeze(0).t(),hcn

class trainer():
    
    def __init__(self,net,optimizer,loss_fn):
        self.net=net
        self.optimizer=optimizer
        self.loss_fn=loss_fn
        self.mean=None

    def train_one_epoch(self,data_list,vel_list):
        hcn=[torch.zeros((1,1,self.net.hidden_size),device=vel_list[0].device),torch.zeros((1,1,self.net.hidden_size),device=vel_list[0].device)]
        train_loss=np.zeros(len(data_list))
        mean=torch.zeros(96,device=data_list[0].device)
        for i in range(len(data_list)):
            mean+=data_list[i].mean(1)
            res=self.net(data_list[i],hc=hcn)[0]
            los=self.loss_fn(res,vel_list[i])
            self.optimizer.zero_grad()
            los.backward()
            self.optimizer.step()
            train_loss[i]=float(los)
        self.mean=mean/len(data_list)
        return train_loss
    
    def test(self,test,vel,return_res=0):
        hcn=[torch.zeros((1,1,self.net.hidden_size),device=vel.device),torch.zeros((1,1,self.net.hidden_size),device=vel.device)]
        res=self.net(test,hc=hcn)[0]
        los=self.loss_fn(res,vel)
        mse=float(los)
        if return_res==0:
            return mse,vel.shape[1]
        else:
            x_cc=float(torch.corrcoef(torch.stack([res[0],vel[0]],axis=0))[0,1])
            y_cc=float(torch.corrcoef(torch.stack([res[1],vel[1]],axis=0))[0,1])
            return x_cc,y_cc,mse,res.detach().cpu().numpy()
    
    def test_online(self,raw,vel,wl,return_res=0):
        channel=self.net.input_size
        state0=np.zeros((96,1))
        state1=np.zeros((96,1))
        res=torch.zeros_like(vel)
        time_list=np.zeros(vel.shape[1])
        cun=torch.zeros((self.net.input_size,wl),dtype=torch.float32,device=vel.device)
        c=0
        hcn=[torch.zeros((1,1,self.net.hidden_size),device=vel.device),torch.zeros((1,1,self.net.hidden_size),device=vel.device)]
        inp=raw[:,10000-750*wl-1500:10000-750*wl+1500]
        inp=inp-np.median(inp,axis=0)
        inp,state0=wave_filter(inp,300,30000,'highpass',state0,N=1)
        inp=np.abs(inp)
        inp,state1=wave_filter(inp,12,30000,'lowpass',state1,N=1)
        inp=inp[:,1500::15].mean(1)-self.mean
        cun[:,c]=torch.tensor(inp,dtype=torch.float32)
        c=(c+1)%wl
        for i in range(1,wl-1):
            inp=raw[:,10000-750*wl+1500*i:10000-750*wl+1500*(i+1)]
            inp=inp-np.median(inp,axis=0)
            inp,state0=wave_filter(inp,300,30000,'highpass',state0,N=1)
            inp=np.abs(inp)
            inp,state1=wave_filter(inp,12,30000,'lowpass',state1,N=1)
            inp=inp[:,::15].mean(1)-self.mean
            cun[:,c]=torch.tensor(inp,dtype=torch.float32)
            c=(c+1)%wl
        for i in range(vel.shape[1]):
            inp=raw[:,10000-750*wl+1500*(i+wl-1)-3:10000-750*wl+1500*(i+wl)]
            s=time.time_ns()
            inp=inp-np.median(inp,axis=0)
            inp,state0=wave_filter(inp,300,30000,'highpass',state0,N=1)
            inp=np.abs(inp)
            inp,state1=wave_filter(inp,12,30000,'lowpass',state1,N=1)
            inp=inp[:,::15].mean(1)-self.mean
            cun[:,c]=torch.tensor(inp,dtype=torch.float32)
            inp=cun.mean(1).reshape(channel,-1)
            res[:,i:i+1],hcn=self.net(inp,hc=hcn)
            e=time.time_ns()
            time_list[i]=e-s
            c=(c+1)%wl
        
        los=self.loss_fn(res,vel)
        
        mse=float(los)
        if return_res==0:
            return mse,vel.shape[1]
        else:
            x_cc=float(torch.corrcoef(torch.stack([res[0],vel[0]],axis=0))[0,1])
            y_cc=float(torch.corrcoef(torch.stack([res[1],vel[1]],axis=0))[0,1])
            return x_cc,y_cc,mse,res.detach().cpu().numpy(),time_list
    
    def net_save(self,path):
        torch.save(self.net.state_dict(),path)
        return
        
    def net_load(self,path):
        self.net.load_state_dict(torch.load(path))
        return

def wave_filter(sequence,cof,sf,mod,zi,N=4):
    b, a = signal.butter(N, cof, btype=mod,fs=sf)
    filtedData,zi = signal.lfilter(b, a, sequence,zi=zi)
    return filtedData,zi

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if __name__ == "__main__":
    hz_list=['/home/zju/xgx/dataset/handwriting/ESA/']
    raw_list=['/home/zju/xgx/dataset/handwriting/raw/trial/']
    res_list=['pseudo']
    dataset_list=['0614.mat','0616.mat','0623.mat','0624.mat','0630.mat','0701.mat']
    single_start_path='./single_start_net.pth'
    single_best_path='./single_best_net.pth'
    classify_start_path='./classify_start_net.pth'
    classify_best_path='./classify_best_net.pth'
    break_start_path='./break_start_net.pth'
    break_best_path='./break_best_net.pth'
    ind_start_path='./ind_start_net.pth'
    ind_best_path='./ind_best_net.pth'
    for w in range(len(hz_list)):
        net_model=LSTM_Net
        loss_fn=nn.MSELoss
        wl=4#The unit is 50ms
        hidden_size=512
        optimizer=torch.optim.Adam
        optimizer_kw={'lr':0.001}
        best_end_interval=100
        fit_type='vel'#optional is 'vel' or 'pos'
        
        for q,dataset in enumerate(dataset_list):
            a=scipy.io.loadmat(hz_list[w]+dataset)
            a['bined_spk']=((a['bined_spk'].T-a['bined_spk'].mean(1))).T
            target_num=a['target_hanzi'].shape[1]
            trial_num=a['trial_target'].shape[0]
            net_args=[a['bined_spk'].shape[0],hidden_size,a['trial_velocity'].shape[0]]
            net_kw={}
            
            CC = np.zeros((trial_num,2))
            MSE = np.zeros((trial_num,1))
            prediction = [0 for i in range(trial_num)]
            time_list = [0 for i in range(trial_num)]
            
            ACCURACY = np.zeros((trial_num,1))
            prediction_classify=[0 for i in range(trial_num)]
            
            CC_dual = np.zeros((trial_num,2))
            MSE_dual = np.zeros((trial_num,1))
            prediction_dual = [0 for i in range(trial_num)]
            
            single_loss_fn=loss_fn().cuda()
            # single_loss_fn=dilate_loss
            single_net=net_model(*net_args,**net_kw).cuda()
            single_optimizer=optimizer(single_net.parameters(),**optimizer_kw)
            single_trainer=trainer(single_net,single_optimizer,single_loss_fn)
            single_trainer.net_save(single_start_path)
            
            
            for i_target in range(a['target_hanzi'].shape[1]):
                target_ind = np.where(a['trial_target']-1 == i_target)[0]
                bins_remove = np.concatenate([np.where(a['trial_mask']-1 == target_ind[i])[1] for i in range(len(target_ind))],axis=0)
                trial_velocity_train=np.delete(a['trial_velocity'],bins_remove,axis=1)
                bined_spk_train=np.delete(a['bined_spk'],bins_remove,axis=1)
                break_ind_train=np.delete(a['break_ind'],bins_remove,axis=1)
                trial_mask_train=np.delete(a['trial_mask'],bins_remove,axis=1)
                trial_velocity_train,bined_spk_train,break_ind_train=cut([trial_velocity_train,bined_spk_train,break_ind_train],trial_mask_train[0],lists=1)
                for i in range(len(trial_velocity_train)):
                    if fit_type == 'pos':
                        trial_velocity_train[i]=np.cumsum(trial_velocity_train[i],axis=1)/100
                    trial_velocity_train[i]=torch.tensor(trial_velocity_train[i],dtype=torch.float32).cuda()
                    bined_spk_train[i]=torch.tensor(bined_spk_train[i],dtype=torch.float32).cuda()
                    break_ind_train[i]=torch.tensor((break_ind_train[i])>0,dtype=torch.int64).cuda()              
                
                trial_velocity_test=a['trial_velocity'][:,bins_remove]
                bined_spk_test=a['bined_spk'][:,bins_remove]
                break_ind_test=a['break_ind'][:,bins_remove]
                trial_mask_test=a['trial_mask'][:,bins_remove]
                trial_velocity_test,bined_spk_test,break_ind_test=cut([trial_velocity_test,bined_spk_test,break_ind_test],trial_mask_test[0],lists=1)
                for i in range(len(trial_velocity_test)):
                    if fit_type == 'pos':
                        trial_velocity_test[i]=np.cumsum(trial_velocity_test[i],axis=1)/100
                    trial_velocity_test[i]=torch.tensor(trial_velocity_test[i],dtype=torch.float32).cuda()
                    bined_spk_test[i]=torch.tensor(bined_spk_test[i],dtype=torch.float32).cuda()
                    break_ind_test[i]=torch.tensor((break_ind_test[i])>0,dtype=torch.int64).cuda()
                
                # single model
                single_trainer.net_load(single_start_path)
                mse_best=1e10
                iteration_best=0
                iteration=0
                
                while True:
                    mse=[]
                    length=[]
                    single_trainer.train_one_epoch(bined_spk_train, trial_velocity_train)
                    for i in range(len(trial_velocity_test)):
                        mse1,len1=single_trainer.test(bined_spk_test[i], trial_velocity_test[i])#Pseudo online experiment using test_online method
                        mse.append(mse1)
                        length.append(len1)
                    mse_sum=0
                    len_sum=0
                    for i in range(len(mse)):
                        mse_sum=mse_sum+length[i]*mse[i]
                        len_sum=len_sum+length[i]
                    mse_sum=mse_sum/len_sum
                    if mse_sum<mse_best:
                        mse_best=mse_sum
                        iteration_best=iteration
                        single_trainer.net_save(single_best_path)
                    print('{0}-{1}-single MSE:{2}'.format(i_target,iteration,mse_sum))
                    if iteration-iteration_best>best_end_interval:
                        break
                    iteration=iteration+1
                
                single_trainer.net_load(single_best_path)
                for i in range(len(target_ind)):
                    CC[target_ind[i],0],CC[target_ind[i],1],MSE[target_ind[i],0],prediction[target_ind[i]],time_list[target_ind[i]] = single_trainer.test1(np.load(raw_list[w]+dataset[:4]+'/{0}.npy'.format(target_ind[i])), trial_velocity_test[i],wl,return_res=1)
                   
            create_path=dataset.split('/')[-1].split('.')[0]
            os.makedirs('/home/zju/xgx/result/{0}/{1}'.format(res_list[w],create_path))
            np.save('/home/zju/xgx/result/{0}/{1}/CC.npy'.format(res_list[w],create_path),CC)
            np.save('/home/zju/xgx/result/{0}/{1}/MSE.npy'.format(res_list[w],create_path),MSE)
            np.save('/home/zju/xgx/result/{0}/{1}/prediction.npy'.format(res_list[w],create_path),np.concatenate(prediction,axis=1))
            np.save('/home/zju/xgx/result/{0}/{1}/time.npy'.format(res_list[w],create_path),np.concatenate(time_list))
