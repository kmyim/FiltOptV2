<<<<<<< HEAD
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import dionysus
import pickle
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import time

def EPH_mask(vertex_values, simplices):

    s2v_lst = [[sorted(s), sorted([[vertex_values[v],v] for v in s], key=lambda x: x[0])] for s in simplices]
    f_ord = [dionysus.Simplex(s[0], s[1][-1][0]) for s in s2v_lst] #takes max
    f_ext = [dionysus.Simplex([-1] + s[0], s[1][0][0]) for s in s2v_lst] #takes min

=======
def EPH_mask(vertex_values, simplices):
    
    s2v_lst = [[sorted(s), sorted([[vertex_values[v],v] for v in s], key=lambda x: x[0])] for s in simplices]
    f_ord = [dionysus.Simplex(s[0], s[1][-1][0]) for s in s2v_lst] #takes max 
    f_ext = [dionysus.Simplex([-1] + s[0], s[1][0][0]) for s in s2v_lst] #takes min
    
>>>>>>> ca469790d2768ada496cb7752c9864cbab8c400b
    ord_dict = {tuple(s[0]): s[1][-1][1] for s in s2v_lst}
    ext_dict = {tuple([-1] + s[0]): s[1][0][1] for s in s2v_lst}

    f_ord.sort(key = lambda s: (s.data, len(s)))
    f_ext.sort(key = lambda s: (-s.data, len(s)))

    #computes persistence
    f = dionysus.Filtration([dionysus.Simplex([-1], -float('inf'))] + f_ord + f_ext)
    m = dionysus.homology_persistence(f)

    dgms = [[[], []], [[], []], [[], []], [[], []]] #H0ord, H0ext, H1rel, H1ext

    for i in range(len(m)):
<<<<<<< HEAD

        dim = f[i].dimension()

        if m.pair(i) < i: continue      # skip negative simplices to avoid double counting
=======
        
        dim = f[i].dimension()
        
        if m.pair(i) < i: continue      # skip negative simplices to avoid double counting 
>>>>>>> ca469790d2768ada496cb7752c9864cbab8c400b
        if m.pair(i) != m.unpaired: #should be no unpaired apart from H0 from fictitious -1 vertex
            pos, neg = f[i], f[m.pair(i)]
            if pos.data != neg.data: #off diagonal
                if -1 in pos and -1 in neg:   #rel1
                    dgms[2][0].append(ext_dict[tuple(neg)])
                    dgms[2][1].append(ext_dict[tuple(pos)])

                elif -1 not in pos and -1 not in neg: #ord0
                    dgms[1][0].append(ord_dict[tuple(pos)])
                    dgms[1][1].append(ord_dict[tuple(neg)])

                else:
<<<<<<< HEAD
                    if dim == 0: #H0ext
                        dgms[0][0].append(ord_dict[tuple(pos)])
                        dgms[0][1].append(ext_dict[tuple(neg)])

=======
                    if dim == 0: #H0ext 
                        dgms[0][0].append(ord_dict[tuple(pos)])
                        dgms[0][1].append(ext_dict[tuple(neg)])
                        
>>>>>>> ca469790d2768ada496cb7752c9864cbab8c400b
                    if dim == 1: #H1ext
                        dgms[3][0].append(ext_dict[tuple(neg)])
                        dgms[3][1].append(ord_dict[tuple(pos)])

<<<<<<< HEAD
    return dgms


class ModelB(nn.Module):

    def __init__(self):

        super(ModelB, self).__init__()

=======
    return dgms 

class ModelB(nn.Module):
    
    def __init__(self):
        
        super(ModelB, self).__init__()
        
>>>>>>> ca469790d2768ada496cb7752c9864cbab8c400b
        self.eig_bandwidth = (4,4)
        self.num_filters = 2
        self.max_num_intervals = 5 #max intervals
        self.features = 4
<<<<<<< HEAD

        self.eig_branchA = nn.Sequential(nn.Linear(1, self.eig_bandwidth[0]), nn.ReLU(True), nn.BatchNorm1d(self.eig_bandwidth[0], affine=False), nn.Linear(self.eig_bandwidth[0], self.eig_bandwidth[1]), nn.ReLU(True), nn.BatchNorm1d(self.eig_bandwidth[1], affine=False), nn.Linear(self.eig_bandwidth[1], 1), nn.ReLU(True), nn.BatchNorm1d(1, affine=False))


        self.eig_branchB = nn.Sequential(nn.Linear(1, self.eig_bandwidth[0]), nn.ReLU(True), nn.BatchNorm1d(self.eig_bandwidth[0], affine=False),nn.Linear(self.eig_bandwidth[0], self.eig_bandwidth[1]), nn.ReLU(True), nn.BatchNorm1d(self.eig_bandwidth[1], affine=False),nn.Linear(self.eig_bandwidth[1], 1), nn.ReLU(True), nn.BatchNorm1d(1, affine=False))

=======
        
        self.eig_branchA = nn.Sequential(nn.Linear(1, self.eig_bandwidth[0]), nn.ReLU(True), nn.BatchNorm1d(self.eig_bandwidth[0], affine=False), nn.Linear(self.eig_bandwidth[0], self.eig_bandwidth[1]), nn.ReLU(True), nn.BatchNorm1d(self.eig_bandwidth[1], affine=False), nn.Linear(self.eig_bandwidth[1], 1), nn.ReLU(True), nn.BatchNorm1d(1, affine=False))
        
        
        self.eig_branchB = nn.Sequential(nn.Linear(1, self.eig_bandwidth[0]), nn.ReLU(True), nn.BatchNorm1d(self.eig_bandwidth[0], affine=False),nn.Linear(self.eig_bandwidth[0], self.eig_bandwidth[1]), nn.ReLU(True), nn.BatchNorm1d(self.eig_bandwidth[1], affine=False),nn.Linear(self.eig_bandwidth[1], 1), nn.ReLU(True), nn.BatchNorm1d(1, affine=False))
        
>>>>>>> ca469790d2768ada496cb7752c9864cbab8c400b
        self.project = nn.Conv1d(self.num_filters, self.features*self.num_filters, kernel_size = 2, stride=2, padding=0, groups = self.num_filters, bias=False)
        self.bn1 = nn.BatchNorm1d(4*self.features*self.num_filters, affine=False)

        self.final_fc_branch = nn.Linear(4*self.num_filters * self.features * self.max_num_intervals, 1, bias=True)
<<<<<<< HEAD

    def forward(self, z):
=======
        
    def forward(self, z):        
>>>>>>> ca469790d2768ada496cb7752c9864cbab8c400b
        L = len(z)
        raw_eigenvalues = [sample['eigenvalues'] for sample in z]
        raw_eigenvectorsqs = [sample['eigenvectors_sq'] for sample in z]
        raw_complexes = [sample['simplices'] for sample in z]

        reindex_slices = np.cumsum([0] + [len(s) for s in raw_eigenvalues])

        flat_list_eigs = torch.tensor([item for sublist in raw_eigenvalues for item in sublist], dtype=torch.float).unsqueeze(1)

        eigs = torch.stack((self.eig_branchA(flat_list_eigs).squeeze(1), self.eig_branchB(flat_list_eigs).squeeze(1)), dim=1)
<<<<<<< HEAD

=======
        
>>>>>>> ca469790d2768ada496cb7752c9864cbab8c400b
        reindexed_eigs = [eigs[reindex_slices[i]: reindex_slices[i+1]] for i in range(L)]

        filter_funcs = [torch.matmul(raw_eigenvectorsqs[i], reindexed_eigs[i]).transpose(0,1) for i in range(L)] # L x num_filters x vertices

        x = torch.zeros([L, self.num_filters, 4, self.max_num_intervals, 2], requires_grad = False)

<<<<<<< HEAD

        for k in range(self.num_filters):

            filter_func = [f[k] for f in filter_funcs]
            filter_func_copy = [f.detach().numpy() for f in filter_func]

            pool = multiprocessing.Pool()
            masks = pool.starmap(EPH_mask, zip(filter_func_copy, raw_complexes))
            pool.close()

=======
        
        for k in range(self.num_filters):
            
            filter_func = [f[k] for f in filter_funcs]
            filter_func_copy = [f.detach().numpy() for f in filter_func]
            
            pool = multiprocessing.Pool()
            masks = pool.starmap(EPH_mask, zip(filter_func_copy, raw_complexes))
            pool.close()
            
>>>>>>> ca469790d2768ada496cb7752c9864cbab8c400b
            for i in range(L):
                m = masks[i]
                for j in range(4):
                    dgm_b = filter_func[i][m[j][0]]
                    dgm_d = filter_func[i][m[j][1]]
                    index = torch.sort(dgm_d - dgm_b, descending=True)[1][0:self.max_num_intervals] #sort by persistence
                    num_intervals = len(index)
                    x[i, k, j, 0:num_intervals, 0] += dgm_b[index]
                    x[i, k, j, 0:num_intervals, 1] += dgm_d[index]

<<<<<<< HEAD

        x = x.view(x.size()[0], self.num_filters, 4*self.max_num_intervals * 2)
        x = self.project(x)
        x = F.relu(x)

        x = x.view(x.size()[0], 4* self.num_filters * self.features, self.max_num_intervals) #for batchnorm
        x = self.bn1(x)

        x = x.reshape([x.size()[0], -1])
        x = self.final_fc_branch(x)

        return x


class ModelC1(nn.Module):

    def __init__(self):

        super(ModelC1, self).__init__()

        self.eig_bandwidth = (4,4)
        self.max_num_intervals = 50 #max intervals
        self.samples = 100

        self.eig_branch = nn.Sequential(nn.Linear(1, self.eig_bandwidth[0]),  nn.BatchNorm1d(self.eig_bandwidth[0]), nn.ReLU(True), nn.Linear(self.eig_bandwidth[0], self.eig_bandwidth[1]), nn.BatchNorm1d(self.eig_bandwidth[1]), nn.ReLU(True),  nn.Linear(self.eig_bandwidth[1], 1),  nn.BatchNorm1d(1))

        #self.eig_branch = nn.Sequential(nn.Linear(1, self.eig_bandwidth[0]), nn.ReLU(True), nn.Linear(self.eig_bandwidth[0], self.eig_bandwidth[1]), nn.ReLU(True),  nn.Linear(self.eig_bandwidth[1], 1), nn.ReLU(True))

        
        self.project = nn.Conv1d(1, 2*self.samples, kernel_size = 2, stride=2, padding=0, groups = 1, bias=True)
        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(4)
        self.image_integrate = nn.Conv1d(4, 4, kernel_size = 1, stride = 1, padding=0, groups = 1, bias=False)

        self.final_fc_branch = nn.Linear(4* self.samples, 1, bias=True)

    def forward(self, z):
        L = len(z)
        raw_eigenvalues = [sample['eigenvalues'] for sample in z]
        raw_eigenvectorsqs = [sample['eigenvectors_sq'] for sample in z]
        raw_complexes = [sample['simplices'] for sample in z]

        reindex_slices = np.cumsum([0] + [len(s) for s in raw_eigenvalues])

        flat_list_eigs = torch.tensor([item for sublist in raw_eigenvalues for item in sublist], dtype=torch.float).unsqueeze(1)

        eigs = self.eig_branch(flat_list_eigs)

        reindexed_eigs = [eigs[reindex_slices[i]: reindex_slices[i+1]] for i in range(L)]

        filter_func = [torch.matmul(raw_eigenvectorsqs[i], reindexed_eigs[i]).squeeze(1) for i in range(L)]
        filter_func_copy = [f.detach().numpy() for f in filter_func]


        x = torch.zeros([L, 4, self.max_num_intervals, 2], requires_grad = False)

        pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
        masks = pool.starmap(EPH_mask, zip(filter_func_copy, raw_complexes))
        pool.close()

        for i in range(L):
            m = masks[i]
            for j in range(4):
                dgm_b = filter_func[i][m[j][0]]
                dgm_d = filter_func[i][m[j][1]]
                index = torch.sort(dgm_d - dgm_b, descending=True)[1][0:self.max_num_intervals] #sort by persistence
                num_intervals = len(index)
                x[i, j, 0:num_intervals, 0] += dgm_b[index]
                x[i, j, 0:num_intervals, 1] += dgm_d[index]

        weight = x[:, :, :, 1] - x[:, :, :, 0]

        x = x.view(L, 1, 4*self.max_num_intervals * 2)
        y = self.project(x) #(L,  4*self.max_num_intervals * 2) -> (L,2*self.samples, 4*self.max_num_intervals)
        y = torch.exp(-torch.pow(y,2))
        y = y.view(L, 2*self.samples, 4, self.max_num_intervals)
        y = y.transpose(0,1)
        y = y*weight
        y = y.reshape(2, self.samples, L, 4, self.max_num_intervals)
        z = y[0, :, :, :, :] * y[1, :, :, :, :]#(self.samples,L, 4, self.max_num_intervals)
        z = torch.sum(z, dim = -1) #( self.samples,L, 4)
        z = z.transpose(0,1)
        z = z.transpose(1,2)#(L, 4, self.samples)
        z = self.bn1(z)
        z = self.image_integrate(z)#(L, 4, self.samples)
        z = self.bn2(z)
        #z = F.relu(z)

        z = z.reshape([L, -1])#(L, 4 * self.samples)
        z = self.final_fc_branch(z)

        return z

class ModelC2(nn.Module):

    def __init__(self):

        super(ModelC2, self).__init__()

        self.eig_bandwidth = (4,4)
        self.num_filters = 2
        self.max_num_intervals = 50 #max intervals
        self.samples = 100

        self.eig_branchA = nn.Sequential(nn.Linear(1, self.eig_bandwidth[0]),nn.BatchNorm1d(self.eig_bandwidth[0]), nn.ReLU(True), nn.Linear(self.eig_bandwidth[0], self.eig_bandwidth[1]), nn.BatchNorm1d(self.eig_bandwidth[1]), nn.ReLU(True),  nn.Linear(self.eig_bandwidth[1], 1),  nn.BatchNorm1d(1))


        self.eig_branchB = nn.Sequential(nn.Linear(1, self.eig_bandwidth[0]), nn.BatchNorm1d(self.eig_bandwidth[0]),nn.ReLU(True), nn.Linear(self.eig_bandwidth[0], self.eig_bandwidth[1]), nn.BatchNorm1d(self.eig_bandwidth[1]), nn.ReLU(True), nn.Linear(self.eig_bandwidth[1], 1),  nn.BatchNorm1d(1))

        self.project = nn.Conv1d(self.num_filters, 2*self.samples*self.num_filters, kernel_size = 2, stride=2, padding=0, groups = self.num_filters, bias=True)
        self.bn1 = nn.BatchNorm1d(self.num_filters*4, affine=False)

        self.final_fc_branch = nn.Linear(4*self.num_filters * self.samples, 1, bias=True)

    def forward(self, z):
        L = len(z)
        raw_eigenvalues = [sample['eigenvalues'] for sample in z]
        raw_eigenvectorsqs = [sample['eigenvectors_sq'] for sample in z]
        raw_complexes = [sample['simplices'] for sample in z]

        reindex_slices = np.cumsum([0] + [len(s) for s in raw_eigenvalues])

        flat_list_eigs = torch.tensor([item for sublist in raw_eigenvalues for item in sublist], dtype=torch.float).unsqueeze(1)

        eigs = torch.stack((self.eig_branchA(flat_list_eigs).squeeze(1), self.eig_branchB(flat_list_eigs).squeeze(1)), dim=1)

        reindexed_eigs = [eigs[reindex_slices[i]: reindex_slices[i+1]] for i in range(L)]

        filter_funcs = [torch.matmul(raw_eigenvectorsqs[i], reindexed_eigs[i]).transpose(0,1) for i in range(L)] # L x num_filters x vertices

        x = torch.zeros([L, self.num_filters, 4, self.max_num_intervals, 2], requires_grad = False)


        for k in range(self.num_filters):

            filter_func = [f[k] for f in filter_funcs]
            filter_func_copy = [f.detach().numpy() for f in filter_func]

            pool = multiprocessing.Pool()
            masks = pool.starmap(EPH_mask, zip(filter_func_copy, raw_complexes))
            pool.close()

            for i in range(L):
                m = masks[i]
                for j in range(4):
                    dgm_b = filter_func[i][m[j][0]]
                    dgm_d = filter_func[i][m[j][1]]
                    index = torch.sort(dgm_d - dgm_b, descending=True)[1][0:self.max_num_intervals] #sort by persistence
                    num_intervals = len(index)
                    x[i, k, j, 0:num_intervals, 0] += dgm_b[index]
                    x[i, k, j, 0:num_intervals, 1] += dgm_d[index]

        weight = x[:, :, :, :, 1] - x[:, :, :, :, 0]

        x = x.view(L, self.num_filters, 4*self.max_num_intervals * 2)
        y = self.project(x) #(L, self.num_filters, 4*self.max_num_intervals * 2) -> (L, self.num_filters*2*self.samples, 4*self.max_num_intervals)
        y = torch.exp(-torch.pow(y,2))
        y = y.view(L, self.num_filters, 2*self.samples, 4,self.max_num_intervals)
        y = y.transpose(1,2)
        y = y.transpose(0,1)
        y = y*weight
        y = y.reshape(2, self.samples, L, self.num_filters * 4, self.max_num_intervals)
        z = y[0, :, :,  :, :] * y[1, :, :,  :, :]#( self.samples, L, self.num_filters * 4, self.max_num_intervals)
        z = torch.sum(z, axis = -1) #(self.samples, L , self.num_filters*4)
        z = z.transpose(0,1)
        z = z.transpose(1,2)#(L, self.num_filters*4, self.samples)
        z = self.bn1(z)

        z = z.reshape([z.size()[0], -1])
        z = self.final_fc_branch(z)

        return z
=======
        
        x = x.view(x.size()[0], self.num_filters, 4*self.max_num_intervals * 2)
        x = self.project(x)
        x = F.relu(x)
        
        x = x.view(x.size()[0], 4* self.num_filters * self.features, self.max_num_intervals) #for batchnorm
        x = self.bn1(x)   
        
        x = x.reshape([x.size()[0], -1])
        x = self.final_fc_branch(x)
 
        return x
        
>>>>>>> ca469790d2768ada496cb7752c9864cbab8c400b
