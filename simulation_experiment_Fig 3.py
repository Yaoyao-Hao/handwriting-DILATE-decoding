import os
import numpy as np
import scipy
import torch
import torch.nn as nn
from dilate_loss import*

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

def input_generate(real_vel,num=1,b=1,noise_std=0):
    length=real_vel.shape[1]
    pos=np.concatenate([[[0],[0]],np.cumsum(real_vel,axis=1)],axis=1)
    t=np.random.rand(length-1)*length
    t=np.sort(t)
    pos_=np.zeros_like(pos)
    pos_[:,-1]=pos[:,-1]
    for i in range(len(t)):
        pos_[:,i+1]=(int(t[i])+1-t[i])*pos[:,int(t[i])]+(t[i]-int(t[i]))*pos[:,int(t[i])+1]
    vel=pos_[:,1:]-pos_[:,:-1]
    return torch.tensor(real_vel).to(torch.float32),[torch.tensor(vel).to(torch.float32)]

class trainer():
    def __init__(self,net,optimizer,loss_fn):
        self.net=net
        self.optimizer=optimizer
        self.loss_fn=loss_fn

    def train_one_epoch(self,data_list,vel_list):
        train_loss=np.zeros(len(data_list))
        for i in range(len(data_list)):
            res=self.net(data_list[i]).T
            
            los=self.loss_fn(res,vel_list[i])
            
            self.optimizer.zero_grad()
            los.backward()
            self.optimizer.step()
            train_loss[i]=float(los)
        return train_loss
    
    def test(self,test,vel,return_res=0):
        res=self.net(test).T
        
        los=self.loss_fn(res,vel)
        
        mse=float(los)

        if return_res==0:
            return mse,vel.shape[1]
        else:
            x_cc=float(torch.corrcoef(torch.stack([res[0],vel[0]],axis=0))[0,1])
            y_cc=float(torch.corrcoef(torch.stack([res[1],vel[1]],axis=0))[0,1])
            return x_cc,y_cc,mse,res.detach()
    
    def net_save(self,path):
        torch.save(self.net.state_dict(),path)
        return
        
    def net_load(self,path):
        self.net.load_state_dict(torch.load(path))
        return

if __name__ == "__main__":
    length=300
    daylength=[0,10920, 23940, 36540, 48780, 63180, 74100]
    pt='ESA'
    daylist=['0614','0616','0623','0624','0630','0701']

    t=0
    label_veldata=[]
    for w in range(len(daylist)):
        day=daylist[w]
        a=scipy.io.loadmat('E:/笔迹拟合/dataset/'+pt+'/'+day+'.mat')
        trial_velocity=cut(a['trial_velocity'],a['trial_mask'][0],lists=0)
        k=[]
        for i in range(a['target_hanzi'].shape[1]):
            k.append(trial_velocity[np.where(a['trial_target']==i+1)[0][0]])
        label_veldata.extend(k)


    resample_num=1
    
    best_end_interval=50
    encoder=nn.Linear(2, 96).cuda()
    test_fn=nn.MSELoss().cuda()
    
    decoder=nn.Linear(96, 2).cuda()
    loss_fn=nn.MSELoss().cuda()
    opt=torch.optim.Adam(decoder.parameters(),lr=0.005)
    single_trainer=trainer(decoder,opt,loss_fn)
    
    decoder2=nn.Linear(96, 2).cuda()
    loss_fn2=dilate_loss
    opt2=torch.optim.Adam(decoder2.parameters(),lr=0.005)
    single_trainer2=trainer(decoder2,opt2,loss_fn2)
    
    single_trainer.net_save('./initial.pth')
    
    true_veldata=[]
    inpdata=[]
    for i in range(len(label_veldata)):
        label_vel,true_vel=input_generate(label_veldata[i],num=resample_num)
        label_veldata[i]=label_vel
        true_veldata.extend(true_vel)
    
    label_veldata=[i for i in label_veldata for j in range(resample_num)]
    
    for i in range(len(label_veldata)):
        label_veldata[i]=label_veldata[i].cuda()
        true_veldata[i]=true_veldata[i].cuda()
        inpdata.append(encoder(true_veldata[i].T).detach().cuda())
        inpdata[i]=inpdata[i]+torch.randn(inpdata[i].shape,device=inpdata[i].device)*0*(inpdata[i].var()**0.5)
    
    np.save('./result/label_vel.npy',torch.cat(label_veldata,dim=1).cpu().numpy())
    np.save('./result/real_vel.npy',torch.cat(true_veldata,dim=1).cpu().numpy())
    
    for j in range(6):
        fit_vel_list=[]
        for k in range(5):
            train_vel_list=label_veldata[:k*resample_num*6]+label_veldata[(k+1)*resample_num*6:(j+1)*resample_num*30]
            train_inp_list=inpdata[:k*resample_num*6]+inpdata[(k+1)*resample_num*6:(j+1)*resample_num*30]
            test_vel_list=label_veldata[k*resample_num*6:(k+1)*resample_num*6]
            test_inp_list=inpdata[k*resample_num*6:(k+1)*resample_num*6]

            
            single_trainer.net_load('./initial.pth')
            loss_best=1e10
            iteration_best=0
            iteration=0
            while True:
                single_trainer.train_one_epoch(train_inp_list, train_vel_list)
                mse=[]
                length=[]
                for i in range(len(test_vel_list)):
                    mse1,len1=single_trainer.test(test_inp_list[i], test_vel_list[i])
                    mse.append(mse1)
                    length.append(len1)
                mse_sum=0
                len_sum=0
                for i in range(len(mse)):
                    mse_sum=mse_sum+length[i]*mse[i]
                    len_sum=len_sum+length[i]
                loss=mse_sum/len_sum
                if loss<loss_best:
                    loss_best=loss
                    iteration_best=iteration
                    single_trainer.net_save('./best.pth')
                        
                if iteration-iteration_best>best_end_interval:
                    break
                iteration=iteration+1
            
            single_trainer.net_load('./best.pth')
            for i in range(len(test_vel_list)):
                _,_,_,fit_vel=single_trainer.test(test_inp_list[i], test_vel_list[i],return_res=1)
                fit_vel_list.append(fit_vel)
        
        np.save('./result/fit_vel_MSE_{0}.npy'.format(j),torch.cat(fit_vel_list,dim=1).cpu().numpy())
        
        
    for j in range(6):
        fit_vel_list=[]
        for k in range(5):
            train_vel_list=label_veldata[:k*resample_num*6]+label_veldata[(k+1)*resample_num*6:(j+1)*resample_num*30]
            train_inp_list=inpdata[:k*resample_num*6]+inpdata[(k+1)*resample_num*6:(j+1)*resample_num*30]
            test_vel_list=label_veldata[k*resample_num*6:(k+1)*resample_num*6]
            test_inp_list=inpdata[k*resample_num*6:(k+1)*resample_num*6]

            
            single_trainer2.net_load('./initial.pth')
            loss_best=1e10
            iteration_best=0
            iteration=0
            while True:
                single_trainer2.train_one_turn(train_inp_list, train_vel_list)
                mse=[]
                length=[]
                for i in range(len(test_vel_list)):
                    mse1,len1=single_trainer2.test(test_inp_list[i], test_vel_list[i])
                    mse.append(mse1)
                    length.append(len1)
                mse_sum=0
                len_sum=0
                for i in range(len(mse)):
                    mse_sum=mse_sum+length[i]*mse[i]
                    len_sum=len_sum+length[i]
                loss=mse_sum/len_sum
                if loss<loss_best:
                    loss_best=loss
                    iteration_best=iteration
                    single_trainer2.net_save('./best.pth')
                        
                if iteration-iteration_best>best_end_interval:
                    break
                iteration=iteration+1
            
            single_trainer2.net_load('./best.pth')
            for i in range(len(test_vel_list)):
                _,_,_,fit_vel=single_trainer2.test(test_inp_list[i], test_vel_list[i],return_res=1)
                fit_vel_list.append(fit_vel)
        np.save('./result/fit_vel_DILATE_{0}.npy'.format(j),torch.cat(fit_vel_list,dim=1).cpu().numpy())
            
    