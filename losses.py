from __future__ import print_function

import torch
import torch.nn as nn
import sys
from math import floor, ceil
import torch.nn.functional as F


class UnifiedFocalLoss(nn.Module):
    def __init__(self, alpha, gamma, interval, min_val, max_val):
        super(UnifiedFocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.interval = interval
        self.min_val = min_val
        self.max_val = max_val
        self.interval_list = torch.tensor(self.calculate_interval_boundaries(self.min_val, self.max_val, self.interval),
                                            dtype=torch.float32)         

    def calculate_interval_boundaries(self, min_value, max_value, num_intervals):
        # Calculate the size of each interval
        interval_size = (max_value - min_value) / num_intervals
        
        # Initialize the list of boundaries starting with the minimum value
        boundaries = [min_value + i * interval_size for i in range(num_intervals + 1)][1:]
        
        return boundaries

    def forward(self, prediction_1, prediction_2, label_1, label_2):
        # Encode labels
        label_vector_1 = self.ordinal_encoding(label_1//12) # Since BUF_MAX is 12
        if(label_2 != None):
            label_vector_2 = self.ordinal_encoding(label_2//12)
        
        # Calculate loss for each prediction
        loss1 = self.focal_loss(prediction_1, label_vector_1)

        if(prediction_2 != None):
            loss2 = self.focal_loss(prediction_2, label_vector_2)
            loss = (loss1 + loss2) / 2
        else:
            loss = loss1

        return loss

    def focal_loss(self, predictions, targets):
        # Apply sigmoid to predictions to ensure they are in range [0, 1]
        predictions = torch.sigmoid(predictions)
        # print(predictions.size(), targets.size())
        # Calculate binary cross-entropy loss elementwise
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Cases for loss calculation based on target values
        positive_mask = (targets > 0).float()
        negative_mask = (1 - positive_mask)

        # Calculate modulating factors
        positive_loss = self.alpha * (torch.abs(targets - predictions) ** self.gamma) * bce_loss
        negative_loss = (1 - self.alpha) * (predictions ** self.gamma) * bce_loss

        # Combine the losses
        modulated_loss = positive_mask * positive_loss + negative_mask * negative_loss
        
        # Average the loss
        final_loss = modulated_loss.mean()

        return final_loss


    def ordinal_encoding(self, label):
        label_vector = torch.zeros(self.interval,1, dtype = label.dtype, device=label.device)
        label = label.unsqueeze(-1)
        interval_list = self.interval_list.to(device=label.device, dtype=label.dtype)
        interval_list = interval_list.unsqueeze(0)
        interval_list = interval_list.repeat(label.shape[0], 1)
        # print(interval_list.shape, label.shape)
        is_greater = (label > interval_list)
        difference = -abs(label.float() - interval_list) * is_greater.float()
        label_vector = torch.pow(2.71828, difference) * is_greater.float()

        return label_vector
    
    def ordinal_decoding(self, prediction):
        interval_list = self.interval_list.to(prediction.device)
        index = torch.argmax(prediction, -1, keepdim=False)  # keepdim is False here
        
        # Use the indices to select the corresponding max values from prediction
        # We use torch.gather with index.unsqueeze(1) to align dimensions for gather
        selected_predictions = torch.gather(prediction, 1, index.unsqueeze(1)).squeeze(1)

        # Compute the log term safely by clamping values to avoid log(0) which would result in -inf
        log_term = -torch.log(1 - selected_predictions.clamp(min=0, max=0.999))

        # Ensure interval_list is accessible and broadcastable for the shape [128]
        if len(interval_list) == 1:
            adjusted_intervals = interval_list.expand_as(log_term)
        else:
            # Use the indices to select the corresponding intervals
            adjusted_intervals = interval_list[index]  # index used directly

        # Final computation
        result = log_term + adjusted_intervals

        return result


    def sigmoid(self, x, base=2.71828):
        return 1 / (1 + torch.pow(base, -x))




class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
    
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')

        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)

            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
            
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        # print("Mask Pos Pairs", mask_pos_pairs)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=torch.float32).to(mask_pos_pairs.device), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
