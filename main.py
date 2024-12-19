import os
import argparse
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from losses import *
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from dataset import DSEDataset
from model import MLPRecommender, TransformRecommender, EarlyStopping
import utils as util
from tensorboardX import SummaryWriter 
from datetime import datetime
from configs import parser


args = parser.parse_args()

if args.which_gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpus
    use_cuda = not args.no_cuda and torch.cuda.is_available()   
    device = torch.device("cuda:0" if use_cuda else "cpu")

def set_seed(RANDOM_SEED):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(RANDOM_SEED) 
    torch.cuda.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
if args.seed is not None:
    RANDOM_SEED = args.seed
    set_seed(RANDOM_SEED)    

if args.test:
    assert args.load_chkpt is not None


# Model parameters
batch_size = args.batch_size
test_batch_size = args.test_batch_size
epochs = args.epoch
embedding_size = args.embedding_size
num_heads = args.num_heads
num_layers = args.num_layers
hidden_nodes_list = [int(item) for item in args.hidden_nodes.split('_')]
feature_dim = args.feature_dim

init_learning_rate = args.lr
alpha = args.alpha

# Saving filename/filepath
tensorboard_file_name = "./tboard_logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# os.system('rm ' + tensorboard_file_name + '/*')
writer = SummaryWriter(tensorboard_file_name)
model_prefix = ''
if args.model == 'MLP':
    model_prefix = 'model={}_alpha={}_embed={}_hidden={}_lr={}_batchsize={}_epoch={}_seed={}'.format(args.model, args.alpha, args.embedding_size, args.hidden_nodes, args.lr, args.batch_size, args.epoch, args.seed) 
    
if args.model == 'Transformer':
    model_prefix = 'model={}_alpha={}_embed={}_head={}_layer={}_hidden={}_lr={}_batchsize={}_epoch={}_seed={}'.format(args.model, args.alpha, args.embedding_size, args.num_heads, args.num_layers, args.hidden_nodes, args.lr, args.batch_size, args.epoch, args.seed) 
model_filename = tensorboard_file_name + '/' + model_prefix + '_model.pth'

def load_dse_dataset(filepath):      
    dataset = pd.read_csv(filepath)    

    train_indices, test_indices = train_test_split(range(len(dataset)), train_size=0.95, random_state=RANDOM_SEED)
    assert len(set(train_indices).intersection(set(test_indices))) == 0  , 'invalid training/testing split'   
    print(f'train_data size: {len(train_indices)}, test_data size: {len(test_indices)}')
    
    train_dataset = DSEDataset(filepath, indices=train_indices, rewards=args.enable_surrogate)
    test_dataset  = DSEDataset(filepath, indices=test_indices, rewards=args.enable_surrogate)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4, drop_last=True)    
        
    return train_loader, test_loader, train_dataset.max_input_features, train_dataset.num_classes

def train(model, device, train_loader, criterion, optimizer, epoch):         
    # set printing functions
    supcon_losses = util.AverageMeter('SUPCON Loss:', ':6.5f')

    if args.enable_surrogate:
        mse_losses = util.AverageMeter('MSE Loss:', ':6.5f')
        losses_list = [supcon_losses, mse_losses]

    else:
        assert "Surrogate is Needed"
        
    progress = util.ProgressMeter(
                    len(train_loader),
                    losses_list,
                    prefix="Epoch: [{}]".format(epoch+1)
                    )
    model.train()
    
    correct = 0
    total = 0
    total_loss = 0
    total_supcon_loss = 0 
    total_mse_loss = 0 
    supcon_loss = 0
    mse_loss = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):     
        if args.enable_surrogate:
            hw_labels, obj_labels = labels
            inputs, hw_labels, obj_labels = inputs.to(device), hw_labels.to(device), obj_labels.to(device)

            hw_outputs, obj_outputs = model(inputs)

            # SupCon Loss requires 3 dimensions, axis=1 has views
            hw_outputs = hw_outputs.unsqueeze(1)

            supcon_loss = criterion[0](hw_outputs, hw_labels) 
            mse_loss = criterion[1](obj_outputs, obj_labels) 
            
        loss = supcon_loss + mse_loss
                
        optimizer.zero_grad()                       
        loss.backward()
        optimizer.step()
        
        # record loss
        total += hw_labels.size(0)
        total_supcon_loss += supcon_loss.item() * hw_labels.size(0) 
        total_mse_loss += mse_loss.item() * obj_labels.size(0) 
        total_loss += total_supcon_loss + total_mse_loss
        
        if batch_idx % 50 == 49:   
            progress.display(batch_idx)

        # update printing information
        supcon_losses.update(supcon_loss.item(), hw_labels.size(0))
        mse_losses.update(mse_loss.item(), obj_labels.size(0))
        
    print('Epoch {}: Training SUPCON Loss = {:.5f}, Training MSE Loss = {:.5f}'.format(epoch+1, total_supcon_loss/total, total_mse_loss/total, 100*correct/total))

    return total_loss/total, total_supcon_loss/total, total_mse_loss/total

def main():    
    train_loader, test_loader, max_input_features, num_classes = load_dse_dataset(args.data)
    
    # Prepare the model
    if args.model == 'MLP':
        model = MLPRecommender(max_input_features, num_classes, embedding_size, hidden_nodes_list).to(device)
    elif args.model == 'Transformer':
        model = TransformRecommender(max_input_features, num_classes, feature_dim, embedding_size, hidden_nodes_list, num_layers, num_heads, enable_surrogate=args.enable_surrogate, surrogate_model=args.surrogate_model).to(device)  

    if args.load_chkpt is not None:
        model.load_state_dict(torch.load(args.load_chkpt, map_location=device)) 
        
    set_seed(RANDOM_SEED)
    
    criterion = [SupConLoss().to(device), nn.L1Loss().to(device)] # Refer to VAESA, use L1 instead of MSE
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)      
    # early_stopper = EarlyStopping(monitor='val_accuracy', patience=5, model_path=model_filename, verbose=1)
    # start training
    for epoch in range(epochs):  
        if args.enable_surrogate:
            train_loss, train_supcon_loss, train_mse_loss = train(model, device, train_loader, criterion, optimizer, epoch)
        
        writer.add_scalars('Loss', {'Train':train_loss}, epoch)
        if args.enable_surrogate:
            writer.add_scalars('SUPCON Loss', {'Train':train_supcon_loss}, epoch)
            writer.add_scalars('MSE Loss', {'Train':train_mse_loss}, epoch)
        
        # if early_stopper.early_stop(test_acc, model):             
        #     break
        
    writer.close()
    print('Finished Training')

if __name__ == "__main__":
    main()

