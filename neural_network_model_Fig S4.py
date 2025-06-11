import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTM_Net(nn.Module):
    
    def __init__(self,input_size,hidden_size,output_size):
        super(LSTM_Net,self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.lstm=nn.LSTM(input_size,hidden_size,1,batch_first=True)
        self.fc1=nn.Linear(hidden_size, output_size)
        
    def forward(self,x,noise=0):
        res=x.t()
        if noise!=0:
            res=res+noise*res.std(dim=0)*torch.randn(res.shape,device=res.device)
        res,_=self.lstm(res.unsqueeze(0))
        res=self.fc1(F.elu(res))
        return res.squeeze(0).t()
    
class Transformer_Net(nn.Module):
    
    def __init__(self,input_size,hidden_size,output_size):
        super(Transformer_Net,self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.Transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=input_size,nhead=6),num_layers=3)
        self.fc1=nn.Linear(input_size, output_size)
        
    def forward(self,x,noise=0):
        res=x.t()
        res=self.Transformer(res.unsqueeze(0))
        res=self.fc1(F.elu(res))
        return res.squeeze(0).t()

class GRU_Net(nn.Module):
    
    def __init__(self,input_size,hidden_size,output_size):
        super(GRU_Net,self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.gru=nn.GRU(input_size,hidden_size,1)
        self.fc1=nn.Linear(hidden_size,output_size)
        
    def forward(self,x,noise=0):
        res=x.t()
        res,_=self.gru(res.unsqueeze(0))
        res=self.fc1(F.elu(res))
        return res.squeeze(0).t()
    
class RNN_Net(nn.Module):
    
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN_Net,self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.rnn=nn.RNN(input_size,hidden_size,1)
        self.fc1=nn.Linear(hidden_size,output_size)
        
    def forward(self,x,noise=0):
        res=x.t()
        res,_=self.rnn(res.unsqueeze(0))
        res=self.fc1(F.elu(res))
        return res.squeeze(0).t()

class TCN(nn.Module):
    def __init__(self,input_size,hidden_size, num_layers=3, kernel_size=2):
        super(TCN, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else hidden_size
            out_channels = hidden_size
            
            pad = nn.ZeroPad2d(padding=((kernel_size - 1) * dilation, 0, 0, 0))
            self.tcn_layers.append(pad)

            tcn_layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                                  stride=1, 
                                  dilation=dilation)
            self.tcn_layers.append(tcn_layer)

    def forward(self, x):
        # x.shape = (batch_size, time_steps, input_size)
        x = x.permute(0, 2, 1)  # (batch_size, input_size, time_steps)
        for tcn_layer in self.tcn_layers:
            x = F.relu(tcn_layer(x))
        x = x.permute(0, 2, 1)
        return x

class TCN_Net(nn.Module):
    
    def __init__(self,input_size,hidden_size,output_size):
        super(TCN_Net,self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.lstm=TCN(input_size,hidden_size,1)
        self.fc1=nn.Linear(hidden_size,output_size)
        
    def forward(self,x,noise=0):
        res=x.t()
        res=self.lstm(res.unsqueeze(0))
        res=self.fc1(F.elu(res))
        return res.squeeze(0).t()

class trainer():
    
    def __init__(self,net,optimizer,loss_fn):
        self.net=net
        self.optimizer=optimizer
        self.loss_fn=loss_fn
        
    def train_one_epoch(self,data_list,vel_list,noise=0):
        train_loss=np.zeros(len(data_list))
        for i in range(len(data_list)):
            res=self.net(data_list[i],noise=noise)
            los=self.loss_fn(res,vel_list[i])
            self.optimizer.zero_grad()
            los.backward()
            self.optimizer.step()
            train_loss[i]=float(los)
        return train_loss
    
    def test(self,test,vel,return_res=0):
        res=self.net(test)
        los=self.loss_fn(res,vel)
        
        mse=float(los)
        if return_res==0:
            return mse,vel.shape[1]
        else:
            x_cc=float(torch.corrcoef(torch.stack([res[0],vel[0]],axis=0))[0,1])
            y_cc=float(torch.corrcoef(torch.stack([res[1],vel[1]],axis=0))[0,1])
            return x_cc,y_cc,mse,res.detach().cpu().numpy()
    
    def net_save(self,path):
        torch.save(self.net.state_dict(),path)
        return
        
    def net_load(self,path):
        self.net.load_state_dict(torch.load(path))
        return
