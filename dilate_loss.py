import torch
import shape_loss
import time_loss

def dilate_loss(outputs, targets, alpha=0.5, gamma=0.01):
    outputs=outputs.t()
    outputs=outputs/outputs.var(0)**0.5
    targets=targets.t()
    targets=targets/targets.var(0)**0.5
    N_output,dimension = outputs.shape
    loss_shape = 0
    softdtw_batch = shape_loss.SoftDTWBatch.apply
    D = torch.zeros((1, N_output,N_output )).to(outputs.device)
    for k in range(1):
        Dk = shape_loss.pairwise_distances(targets[:,:].view(-1,dimension),outputs[:,:].view(-1,dimension))
        D[k:k+1,:,:] = Dk     
    loss_shape = softdtw_batch(D,gamma)
    
    path_dtw = time_loss.PathDTWBatch.apply
    path = path_dtw(D,gamma)
    Omega =  shape_loss.pairwise_distances(torch.arange(1,N_output+1).view(N_output,1)).to(outputs.device)
    loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 
    loss = alpha*loss_shape+ (1-alpha)*loss_temporal
    return loss

def get_path_matrix(outputs, targets, alpha=0.5, gamma=0.01):
    outputs=outputs.t()
    outputs=outputs/outputs.var(0)**0.5
    targets=targets.t()
    targets=targets/targets.var(0)**0.5
    N_output,dimension = outputs.shape
    loss_shape = 0
    softdtw_batch = shape_loss.SoftDTWBatch.apply
    D = torch.zeros((1, N_output,N_output )).to(outputs.device)
    for k in range(1):
        Dk = shape_loss.pairwise_distances(targets[:,:].view(-1,dimension),outputs[:,:].view(-1,dimension))
        D[k:k+1,:,:] = Dk     
    loss_shape = softdtw_batch(D,gamma)
    
    path_dtw = time_loss.PathDTWBatch.apply
    path = path_dtw(D,gamma)
    return path