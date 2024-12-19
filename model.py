import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, max_input_features, embedding_size):
        """
        Inputs:
            max_input_features - a list, maximum values for each feature
            embedding_size - embedding size for each feature
        """
        super(EmbeddingLayer, self).__init__()
        self.max_input_features = max_input_features
        self.num_input_features = len(max_input_features)
        
        self.embedding_size = embedding_size
        embedding_layer = []
        for i in range (len(self.max_input_features)):
            embedding_layer.append(nn.Embedding(self.max_input_features[i]+1, self.embedding_size))
        self.embed = nn.ModuleList(embedding_layer)
    
    def forward(self, x):
        for i, embed_layer in enumerate(self.embed):
            if i == 0:
                out = embed_layer(x[:, i]).unsqueeze(2)
            else:                
                out = torch.cat((out, embed_layer(x[:, i]).unsqueeze(2)), 2)   
                
        out = out.permute(0,2,1)
        
        return out
    
class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=None, dropout=0.0, feedforward=True):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP. Usually, the inner dimensionality of the MLP is 2-8 larger than the dimensionality of the original input 
            dropout - Dropout probability to use in the dropout layers
            feedforward - if activate feedforward network
        """
        super(EncoderBlock, self).__init__()

        self.feedforward = feedforward
        self.self_attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True) 
        
        if dim_feedforward == None:
            self.dim_feedforward = input_dim*2
        else:
            self.dim_feedforward = dim_feedforward

        if self.feedforward:
            # Two-layer MLP
            self.linear_net = nn.Sequential(
                nn.Linear(input_dim, self.dim_feedforward),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.Linear(self.dim_feedforward, input_dim)
            )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention part
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        if self.feedforward:
            # MLP part
            linear_out = self.linear_net(x)
            x = x + self.dropout(linear_out)
            x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        """
        Inputs:
            num_layers - number of EncoderBlock
        """
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps
  
class TransformRecommender(nn.Module):
    def __init__(self, max_input_features, num_classes, feature_dim, embedding_size, hidden_nodes_list, num_layers=1, num_heads=1, dropout=0.0, enable_surrogate=False, surrogate_model='orig'):
        super(TransformRecommender, self).__init__()
        self.max_input_features = max_input_features
        self.num_input_features = len(max_input_features)
        self.embedding_size = embedding_size
        
        self.transformer_output_dim = int(self.embedding_size*self.num_input_features)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.feature_dim = feature_dim
        
        self.hidden_nodes_list = hidden_nodes_list
        self.num_classes = num_classes
        
        
        # Embedding layer
        self.embed = EmbeddingLayer(max_input_features=self.max_input_features, 
                                    embedding_size=self.embedding_size)        
        
        # Transformer
        self.transformer = TransformerEncoder(num_layers=self.num_layers,
                                              input_dim=self.embedding_size,
                                              num_heads=self.num_heads,
                                              dim_feedforward=4*self.embedding_size,
                                              dropout=self.dropout, 
                                              feedforward=True)
        
        layer = []
        for i in range(len(self.hidden_nodes_list)):
            if i == 0:
                layer.append(nn.Linear(self.transformer_output_dim, self.hidden_nodes_list[i]))
            else:
                layer.append(nn.Linear(self.hidden_nodes_list[i-1], self.hidden_nodes_list[i]))
            layer.append(nn.Dropout(dropout, inplace=True))          
            layer.append(nn.ReLU(inplace=True))              
        layer.append(nn.Linear(self.hidden_nodes_list[-1], self.feature_dim))   
        self.output_net = nn.Sequential(*layer)

        self.enable_surrogate = enable_surrogate
        self.surrogate_model = surrogate_model
        if self.enable_surrogate:
            if self.surrogate_model == 'orig':
                self.surrogate  = nn.Sequential(
                        nn.Linear(int(self.embedding_size*self.num_input_features), 512), 
                        nn.Tanh(), 
                        nn.Linear(512, 1),
                        )
            if self.surrogate_model == 'deep':
                self.surrogate  = nn.Sequential(
                        nn.Linear(int(self.embedding_size*self.num_input_features), 512), 
                        nn.Tanh(), 
                        nn.Linear(512, 256), 
                        nn.Tanh(), 
                        nn.Linear(256, 1),
                        )
            
    def forward(self, x):
        x = self.embed(x)   
        x1 = self.transformer(x)   
        x1 = x1.reshape(x1.size(0), -1)
        x1 = self.output_net(x1)

        if self.enable_surrogate:
            x = x.reshape(x.size(0), -1)
            x2 = self.surrogate(x)     
            return x1, x2
    
class MLPRecommender(nn.Module):
    def __init__(self, max_input_features, num_classes, embedding_size, hidden_nodes_list):
        super(MLPRecommender, self).__init__()
        self.max_input_features = max_input_features
        self.num_input_features = len(max_input_features)
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        
        self.embed = EmbeddingLayer(max_input_features, embedding_size)
        
        self.hidden_nodes_list = hidden_nodes_list
        layer = []
        for i in range(len(self.hidden_nodes_list)):
            if i == 0:
                layer.append(nn.Linear(int(self.embedding_size*self.num_input_features), self.hidden_nodes_list[i]))
            else:
                layer.append(nn.Linear(self.hidden_nodes_list[i-1], self.hidden_nodes_list[i]))  
            layer.append(nn.BatchNorm1d(self.hidden_nodes_list[i]))         
            layer.append(nn.ReLU(inplace=True))      
        layer.append(nn.Linear(self.hidden_nodes_list[-1], self.num_classes))     
        self.classifier = nn.Sequential(*layer)
    
    def forward(self, x): # [bs, num_input_features]
        x = self.embed(x)   
        x = x.reshape(x.size(0), -1)   
        x = self.classifier(x)     
        return x

class SimpleMLP(nn.Module): # no embedding layer 
    def __init__(self, max_input_features, num_classes, hidden_nodes_list):
        super(SimpleMLP, self).__init__()
        self.max_input_features = max_input_features
        self.num_input_features = len(max_input_features)
        self.num_classes = num_classes
        
        self.hidden_nodes_list = hidden_nodes_list
        layer = []
        for i in range(len(self.hidden_nodes_list)):
            if i == 0:
                layer.append(nn.Linear(int(self.num_input_features), self.hidden_nodes_list[i]))
            else:
                layer.append(nn.Linear(self.hidden_nodes_list[i-1], self.hidden_nodes_list[i]))  
            layer.append(nn.BatchNorm1d(self.hidden_nodes_list[i]))         
            layer.append(nn.ReLU(inplace=True))      
        layer.append(nn.Linear(self.hidden_nodes_list[-1], self.num_classes))     
        self.classifier = nn.Sequential(*layer)
    
    def forward(self, x): # [bs, num_input_features]
        x = self.classifier(x)     
        return x

class EarlyStopping:
    """Early stops the training if validation acc/loss doesn't improve after a given patience."""
    def __init__(self, monitor='val_accuracy', patience=1, delta=0, model_path='checkpoint.pt', verbose=0):
        """
        Args:
            patience (int): How long to wait after last time validation acc/loss improved.
            verbose (bool): If True, prints a message for each validation acc/loss improvement. 
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.monitor = monitor
        self.patience = patience
        self.counter = 0
        self.best_score = 0 if self.monitor == 'val_accuracy' else np.inf
        self.delta = delta
        self.model_path = model_path
        self.verbose = verbose

    def early_stop(self, score, model):
                    
        if self.monitor == 'val_accuracy':
            if score < self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}, Best {self.monitor} = {self.best_score:.3f}')
                if self.counter >= self.patience:
                    return True
                return False
            else:
                self.save_checkpoint(score, model)
                self.best_score = score
                self.counter = 0
                return False
                            
        if self.monitor == 'val_loss':
            if score > self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}, Best {self.monitor} = {self.best_score:.3f}')
                if self.counter >= self.patience:
                    return True
                return False
            else:
                self.save_checkpoint(score, model)
                self.best_score = score
                self.counter = 0
                return False
    
    def save_checkpoint(self, score, model):
        if self.verbose:
            if self.monitor == 'val_accuracy':
                print(f'Validation accuracy increased ({self.best_score:.3f} --> {score:.3f}).  Saving model to {self.model_path}')
            if self.monitor == 'val_loss':
                print(f'Validation loss decreased ({self.best_score:.3f} --> {score:.3f}).  Saving model to {self.model_path}')
        torch.save(model.state_dict(), self.model_path)