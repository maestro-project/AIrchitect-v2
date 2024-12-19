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
import sys
from dataset import DSEDataset
from model import MLPRecommender, TransformRecommender, EarlyStopping
import utils as util
from networks import LinearClassifier
from networks import DecoderBlock
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
    model_prefix = 'model={}_alpha={}_embed={}_hidden={}_lr={}_batchsize={}_epoch={}_seed={}'.format( args.model, args.alpha, args.embedding_size, args.hidden_nodes, args.lr, args.batch_size, args.epoch, args.seed) 
    
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


def train(model, classifier, device, train_loader, criterion, optimizer, epoch):         
    # set printing functions
    ce_losses = util.AverageMeter('CE Loss:', ':6.5f')
    top1 = util.AverageMeter('Acc(%):', ':6.2f')
    losses_list = [ce_losses, top1]
        
    progress = util.ProgressMeter(
                    len(train_loader),
                    losses_list,
                    prefix="Epoch: [{}]".format(epoch+1)
                    )
    model.eval()
    classifier.train()
    
    correct = 0
    total = 0
    total_loss = 0
    total_ce_loss = 0 


    for batch_idx, (inputs, labels) in enumerate(train_loader):     
        hw_labels, obj_labels = labels
        inputs, hw_labels, obj_labels = inputs.to(device), hw_labels.to(device), obj_labels.to(device)

        hw_outputs = classifier(model(inputs)[0])
        if(args.ordinal == True):
            ce_loss = criterion[-2](hw_outputs, None, hw_labels, None)
        
        else:
            ce_loss = criterion[-1](hw_outputs, hw_labels)
            
        loss = ce_loss
                
        optimizer.zero_grad()                       
        loss.backward()
        optimizer.step()
        
        # measure accuracy and record loss
        if(args.ordinal == False):
            _, pred = torch.max(hw_outputs.data, 1)
            acc = 100 * (pred == hw_labels).sum().item() / hw_labels.size(0)
            correct += (pred == hw_labels).sum().item()
        
        else:
            # Add ordinal decoding here
            pred = criterion[-2].ordinal_decoding(hw_outputs.data)
            # acc = 100 * (pred == (hw_labels//12)).sum().item() / hw_labels.size(0)
            # correct += (pred == (hw_labels//12)).sum().item()
            correct_predictions = (pred >= (hw_labels // 12) - 5) & (pred <= (hw_labels // 12) + 5)

            # Calculate accuracy as the percentage of correct predictions.
            acc = 100 * correct_predictions.sum().item() / hw_labels.size(0)

            # Update the total number of correct predictions.
            correct += correct_predictions.sum().item()
        total += hw_labels.size(0) 
        total_ce_loss += ce_loss.item() * hw_labels.size(0) 
        
        if batch_idx % 50 == 49:   
            progress.display(batch_idx)

        # update printing information
        ce_losses.update(ce_loss.item(), hw_labels.size(0))
        top1.update(acc, hw_labels.size(0))
        
    print('Epoch {}: Training CE Loss = {:.5f}, Training Accuracy = {:.2f}'.format(epoch+1, total_ce_loss/total, 100*correct/total))
    return 100*correct/total, total_loss/total

def main():    
    train_loader, test_loader, max_input_features, num_classes = load_dse_dataset(args.data)
    
    # Prepare the model
    if args.model == 'MLP':
        model = MLPRecommender(max_input_features, num_classes, embedding_size, hidden_nodes_list).to(device)
    elif args.model == 'Transformer':
        model = TransformRecommender(max_input_features, num_classes, feature_dim, embedding_size, hidden_nodes_list, num_layers, num_heads, enable_surrogate=args.enable_surrogate, surrogate_model=args.surrogate_model).to(device)  

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    if(args.classifier == "Linear"):
        classifier = LinearClassifier(name='resnet50', feat_dim = feature_dim, num_classes=(args.interval if args.ordinal else num_classes)).to(device)

    else:
        print("Transformer Running")
        classifier = DecoderBlock(input_dim = feature_dim, num_heads=feature_dim, num_classes=(args.interval if args.ordinal else num_classes)).to(device)

    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"Total number of parameters: {total_params}")

    # sys.exit()
    if args.load_chkpt is not None:
        print("Loading")
        model.load_state_dict(torch.load(args.load_chkpt, 
                                            map_location=device))

        # Freeze all the parameters in the model
        for param in model.parameters():
            param.requires_grad = False
        
    set_seed(RANDOM_SEED)
    
    criterion = [SupConLoss().to(device), nn.L1Loss().to(device), UnifiedFocalLoss(alpha=0.75, gamma=1.0, min_val = 0, max_val=63, interval = args.interval).to(device), nn.CrossEntropyLoss().to(device)] # Refer to VAESA, use L1 instead of MSE
    optimizer = torch.optim.Adam(classifier.parameters(), lr=init_learning_rate)      

    # start training
    for epoch in range(epochs):  
        if args.enable_surrogate:
            train_acc, train_loss = train(model, classifier, device, train_loader, criterion, optimizer, epoch)
        
        writer.add_scalars('Loss', {'Train':train_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train':train_acc}, epoch)
        
        # if early_stopper.early_stop(test_acc, model):             
        #     break
        
    writer.close()
    print('Finished Training')

if __name__ == "__main__":
    main()

