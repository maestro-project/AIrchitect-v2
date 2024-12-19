import argparse

#----------------------------
# Argument parser.
#----------------------------

parser = argparse.ArgumentParser(description='AIrchitectv2')
parser.add_argument('--data', type=str, default='./dse_dataset/dataset_100k.csv', required=True, help='input data file')

parser.add_argument('--save', action='store_true', help='save the model')
parser.add_argument('--test', action='store_true', help='test only')

parser.add_argument('--load_chkpt', default=None, help='load saved model')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

parser.add_argument('--which_gpus', '-gpu', type=str, default='0', help='which gpus to use')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')

# Model training settings
parser.add_argument('--model', type=str, default='Transformer', choices=['Transformer', 'MLP', 'SVM-Linear', 'SVM-RBF'], help='select the model for recommendation')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=1000, help='input batch size for testing (default: 1000)')
parser.add_argument('--epoch', type=int, default=60, help='number of epochs to train (default: 60)')
parser.add_argument('--embedding_size', type=int, default=32, help='embedding output size (default: 32)')

# for recommender
parser.add_argument('--num_heads', type=int, default=4, help='number of heads for self-attention layer (default: 1)')
parser.add_argument('--num_layers', type=int, default=4, help='number of layers for encoder block (default: 1)')
parser.add_argument('--hidden_nodes', type=str, default="512_512", help='hidden_nodes_list after embedding layers (default: 512_512)')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')

# for surrogate
parser.add_argument('--enable_surrogate', action='store_true', default=False, help='whether to train a performance predictor and a classifier at the same time (default: False)')
parser.add_argument('--surrogate_model', default='deep', help='set predictor model. options [orig, deep]')
parser.add_argument('--alpha', type=float, default=0.2, help='weighting ratio for the loss of the surrogate (default: 0.2)')

# Supervised/Unsupervised contrastive learning
parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

# Contrastive Loss Embedding Dimension
parser.add_argument('--feature_dim', type=int, default=128, help='Select Embedding Dimension for Evaluating Contrastive Loss')
# temperature
parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
# other setting
parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')

# Combine regression and classification
parser.add_argument('--ordinal', action='store_true',
                        help='using Combined Reg and Cla')

parser.add_argument('--interval', type=int, default=16)


parser.add_argument('--classifier', type=str, default='Linear')

                