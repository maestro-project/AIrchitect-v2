from torch.utils.data.dataset import Dataset
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer, StandardScaler

class DSEDataset(Dataset):
    def __init__(self, filepath, indices=None, rewards=False):
        self.rewards = rewards      
        
        self.dataset_df = pd.read_csv(filepath)    
        self.x_data = self.dataset_df[['K','C','X','Y','R','S','df']].to_numpy()        
        self.num_classes = 64*12
        
        # dimensions transformation
        self.x_data[:, 0] = self.x_data[:, 0]//2 - 1    # random.randrange(2, 2*256+1, 2) -> 0~255
        self.x_data[:, 1] = self.x_data[:, 1]//2 - 1    # random.randrange(2, 2*256+1, 2) -> 0~255
        self.x_data[:, 2] = self.x_data[:, 2]//8        # random.choice([1] + [i for i in range(8, 8*32+1, 8)]) -> 0~32
        self.x_data[:, 3] = self.x_data[:, 3]//8        # random.choice([1] + [i for i in range(8, 8*32+1, 8)]) -> 0~32
        self.x_data[:, 4] = (self.x_data[:, 4] - 1) // 2 # [1,3,5,7,9]   -> 0~4
        self.x_data[:, 5] = (self.x_data[:, 5] - 1) // 2 # [1,3,5,7,9]   -> 0~4
        self.max_input_features = [255,255,32,32,4,4,2] 

        # df transformation
        for i in range(self.x_data.shape[0]):
            df = self.x_data[i][6]
            if df == 'dla': self.x_data[i][6] = 0
            elif df == 'eye': self.x_data[i][6] = 1
            elif df == 'shi': self.x_data[i][6] = 2     
                
        # REWARDS transformation
        self.rewards_data = self.dataset_df[['rewards']].to_numpy()
        # Refer to VAESA
        self.rewards_data = np.log(self.rewards_data)
        self.rewards_data_mean = self.rewards_data.mean()
        self.rewards_data_std = self.rewards_data.std()
        self.rewards_data = (self.rewards_data - self.rewards_data_mean) / self.rewards_data_std
        
        self.y_data = self.dataset_df[['ConfigID']].to_numpy().squeeze(1)
        
            
        self.x_data = self.x_data.astype(dtype='int')
        self.y_data = self.y_data.astype(dtype='int')
        self.rewards_data = self.rewards_data.astype(dtype='float32')
        
        if indices is not None:            
            self.x_data = self.x_data[indices]
            self.y_data = self.y_data[indices]    
            self.rewards_data = self.rewards_data[indices] 
        
        assert len(self.x_data) == len(self.y_data), 'mismatched length!'
        self.dataset_len = len(self.x_data)
        
    def __getitem__(self, index):
        if self.rewards:
            return (torch.from_numpy(self.x_data[index]), (self.y_data[index], self.rewards_data[index]))        
        else:
            return (torch.from_numpy(self.x_data[index]), self.y_data[index])

    def __len__(self):
        return self.dataset_len
        
    def get_max_input_features(self):
        return self.max_input_features
