import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import dionysus
import pickle

import numpy as np
#from dgm2path_utils import *
from sig_model_V2 import *

import argparse

parser = argparse.ArgumentParser(description='vectorisation options')
parser.add_argument("--s0",  default=4, type=int, help="Number of line projections for the H0 dgms")
parser.add_argument("--s1",  default=4, type=int, help="Number of line projections for the H1 dgms")
parser.add_argument("--m0", default=3, type=int, help="Signature level for H0 U H1rel dgm.")
parser.add_argument("--m1", default=3, type=int, help="Signature level for H1ext dgm.")
parser.add_argument("--do", default=0.5, type=float, help="Dropout rate for final fully connected layers.")
parser.add_argument("--fi", default=4, type=int, help="Number of filtrations")

args = parser.parse_args()


ef_num_hlayers = 4
ef_num_hu = 16
ef_num_filters = args.fi
num_slices = [args.s0, args.s1]
pslevel = [args.m0, args.m1]
dropout = args.do
max_epoch = 400

savestr = str(args.s0) + '_' + str(args.s1) + '_' + str(args.m0) + '_' + str(args.m1) + '_' + '%.0f'%(args.do*100) + '_' + str(max_epoch)

datasubdir = 'DHFR_data/'
expt_dump = 'expt_results/'

labels = pickle.load(open(datasubdir + 'labels.pkl', 'rb'))
Ndp = len(labels)
label_matrix = np.zeros([Ndp])
for i in range(Ndp):
    if labels[i] == 1:
        label_matrix[i] = 1
label_tensor = torch.from_numpy(label_matrix).float()


print('load data')
all_raw = [pickle.load(open(datasubdir + str(fi) + '_torch.pkl', 'rb')) for fi in range(Ndp)]
arr = list(range(Ndp)) #randomises 90-10 train-val split
np.random.seed(1)
train_acc = []
valid_acc = []


# Train / Test split
cutoff =( Ndp * 9) // 10
batch_size = 68
train_batches = np.ceil((cutoff)/batch_size).astype(int)




print('run model')
# Run mnodel

lambda_lr = lambda epoch: 0.1**(epoch/max_epoch)
criterion = nn.BCEWithLogitsLoss()


for trys in range(10):

    np.random.shuffle(arr)
    print('try = ', trys)

    for fold in range(10):
        print('fold ', fold)
        bottom, top = fold*batch_size, (fold+1)*batch_size

        valid_idx = arr[bottom:top]
        train_idx = arr[0:bottom] + arr[top:]

        train_eval = [all_raw[fi] for fi in train_idx]
        valid_eval = [all_raw[fi] for fi in valid_idx]
        train_eval_labels = label_tensor[train_idx].bool()
        valid_eval_labels = label_tensor[valid_idx].bool()

        torch.manual_seed(999)
        pht = dgmslice_ps2(ef_num_hlayers, ef_num_hu, ef_num_filters, pslevel, num_slices, dropout)

        optimizer  = optim.Adam(pht.parameters(), lr=1e-2, weight_decay = 0.0)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda_lr)

        for epoch in range(max_epoch):

            pht.train()
            np.random.shuffle(train_idx)

            for b in range(train_batches):

                bottom, top = b*batch_size, min((b+1)*batch_size, cutoff)
                inputs = [all_raw[fi] for fi in train_idx[bottom:top]]

                train_labels = label_tensor[train_idx[bottom:top]]
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = pht(inputs)
                loss = criterion(outputs.view(-1), train_labels)
                loss.backward(retain_graph=True)
                optimizer.step()

            scheduler.step()


            if (epoch+1) % 10 == 0:

                pht.eval()
                train_eval_outputs = pht(train_eval)
                valid_eval_outputs = pht(valid_eval)
                train_acc.append(int(((train_eval_outputs.view(-1) > 0) == train_eval_labels).sum())/len(train_eval))
                valid_acc.append(int(((valid_eval_outputs.view(-1) > 0) == valid_eval_labels).sum())/len(valid_eval))

                print(train_acc[-1], valid_acc[-1])


print('Finished Training')
pickle.dump(valid_acc, open(expt_dump + 'sign2_' + savestr + '_valid_acc.pkl', 'wb'))
pickle.dump(train_acc, open(expt_dump + 'sign2_' + savestr + '_train_acc.pkl', 'wb'))
