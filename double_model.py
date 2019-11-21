import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import dionysus
import pickle
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

<<<<<<< HEAD
from Models import ModelC1

print('Finish import ### load labels')
datasubdir = 'DHFR_data/'
expt_dump = 'expt_results/'

print(multiprocessing.cpu_count())
=======
import Models

datasubdir = 'DHFR_data/'
expt_dump = 'expt_results/'

>>>>>>> ca469790d2768ada496cb7752c9864cbab8c400b

# Labels

labels = pickle.load(open(datasubdir + 'labels.pkl', 'rb'))
Ndp = len(labels) 
label_matrix = np.zeros([Ndp])
for i in range(Ndp):
    if labels[i] == 1:
        label_matrix[i] = 1
label_tensor = torch.from_numpy(label_matrix).float()


# Train / Test split
cutoff =( Ndp * 9) // 10 
<<<<<<< HEAD
batch_size = 68
train_batches = np.ceil((cutoff)/batch_size).astype(int)

max_epoch = 300

print('load data')
=======
print(cutoff, Ndp, cutoff/9*10)
batch_size = 68
train_batches = np.ceil((cutoff)/batch_size).astype(int)
print(train_batches, batch_size)
max_epoch = 200

>>>>>>> ca469790d2768ada496cb7752c9864cbab8c400b
all_raw = [pickle.load(open(datasubdir + str(fi) + '_torch.pkl', 'rb')) for fi in range(Ndp)]
arr = list(range(Ndp)) #randomises 90-10 train-val split
np.random.seed(1)
train_acc = []
valid_acc = []

<<<<<<< HEAD
print('run model')
# Run mnodel

lambda_lr = lambda epoch: 0.5**(epoch/50)
=======

# Run mnodel

lambda_lr = lambda epoch: np.exp(np.log(0.1)/max_epoch) ** epoch
>>>>>>> ca469790d2768ada496cb7752c9864cbab8c400b
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
<<<<<<< HEAD
        train_eval_labels = label_tensor[train_idx].bool() 
        valid_eval_labels = label_tensor[valid_idx].bool()

        torch.manual_seed(999)
        pht = ModelC1()

        optimizer  = optim.Adam(pht.parameters(), lr=1e-2, weight_decay=0.0)
=======
        train_eval_labels = label_tensor[train_idx] 
        valid_eval_labels = label_tensor[valid_idx]

        torch.manual_seed(999)
        pht = ModelB()

        optimizer  = optim.Adam(pht.parameters(), lr=1e-3, weight_decay=0.0)
>>>>>>> ca469790d2768ada496cb7752c9864cbab8c400b
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
            
<<<<<<< HEAD
            if (epoch+1) % 30 == 0:
=======
            if (epoch+1) % 20 == 0:
>>>>>>> ca469790d2768ada496cb7752c9864cbab8c400b
                
                pht.eval()
                train_eval_outputs = pht(train_eval)
                valid_eval_outputs = pht(valid_eval)
<<<<<<< HEAD
                train_acc.append(int(((train_eval_outputs.view(-1) > 0) == train_eval_labels).sum())/len(train_eval))
                valid_acc.append(int(((valid_eval_outputs.view(-1) > 0) == valid_eval_labels).sum())/len(valid_eval))
=======
                train_acc.append(int(((train_eval_outputs.view(-1) > 0) == train_eval_labels.byte()).sum())/len(train_eval))
                valid_acc.append(int(((valid_eval_outputs.view(-1) > 0) == valid_eval_labels.byte()).sum())/len(valid_eval))
>>>>>>> ca469790d2768ada496cb7752c9864cbab8c400b

                print(train_acc[-1], valid_acc[-1])

print('Finished Training')
<<<<<<< HEAD
pickle.dump(valid_acc, open(expt_dump + 'epoch300_single_filt_valid_acc.pkl', 'wb'))
pickle.dump(train_acc, open(expt_dump + 'epoch300_single_filt_train_acc.pkl', 'wb'))
=======
pickle.dump(np.reshape(valid_acc, [10, 10, 10]), open(expt_dump + 'epoch200_double_filt_valid_acc.pkl', 'wb'))
pickle.dump(np.reshape(train_acc, [10, 10, 10]), open(expt_dump + 'epoch200_double_filt_train_acc.pkl', 'wb'))
>>>>>>> ca469790d2768ada496cb7752c9864cbab8c400b
