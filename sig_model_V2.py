
import torch
import torch.nn as nn
import numpy as np
import iisignature
from utils import EPH_fast2
import multiprocessing


def linear_block(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Linear(in_f, out_f, *args, **kwargs),
        nn.BatchNorm1d(out_f),
        nn.ReLU()
    )

class SigFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,X, m):
        result=iisignature.sig(X.detach().numpy(), m)
        ctx.save_for_backward(X,torch.tensor(m))
        return torch.tensor(result)

    @staticmethod
    def backward(ctx, grad_output):
        X,m,  = ctx.saved_tensors
        result = iisignature.sigbackprop(grad_output.detach().numpy(),X.detach().numpy(),int(m))

        return torch.tensor(result), None

class eigf(nn.Module):

    def __init__(self, num_hlayers, num_hu):

        super(eigf, self).__init__()

        self.hlayers = num_hlayers
        self.hu = num_hu

        mlp_list = [linear_block(1,self.hu)]
        for i in range(self.hlayers):
            mlp_list.append(linear_block(self.hu, self.hu))
        mlp_list.append(linear_block(self.hu, 1))

        self.func_on_eig = nn.Sequential(*mlp_list)

    def forward(self, evs):

        return self.func_on_eig(evs)



class dgmslice_ps2(nn.Module):

    def __init__(self, ef_num_hlayers, ef_num_hu, ef_num_filters, pslevel, num_slices, dropout):
        '''
        for each diagram, we have different slices
        Across different functions, we have the same parameters for #h0slices, #h0siglevel, #h1slices, #h1siglevel


        '''
        super(dgmslice_ps2, self).__init__()

        self.filters = ef_num_filters
        self.pslevel = pslevel #list [pslevel for h0, pslevel for h1]
        self.slices = num_slices #list [#slices for h0, #slices for pslevel for h1]

        #self.siglength = [iisignature.siglength(num_slices+1, m) for m in pslevel]
        self.siglength = [iisignature.siglength(num_slices[i], pslevel[i]) for i in range(2)] #
        self.eigfs = nn.ModuleList()
        self.projections0 = nn.ModuleList()
        self.projections1 = nn.ModuleList()
        self.final_vec_length = self.filters * sum(self.siglength)


        for i in range(self.filters):
            self.eigfs.append(eigf(ef_num_hlayers, ef_num_hu))

        for i in range(self.filters):
            self.projections0.append(nn.Conv1d(1,self.slices[0],  2, 2))

        for i in range(self.filters):
            self.projections1.append(nn.Conv1d(1,self.slices[1],  2, 2))

        self.final_layer = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.final_vec_length, 1), nn.BatchNorm1d(1))

    def forward(self, mb):

        L = len(mb)
        raw_eigenvalues = [sample['eigenvalues'] for sample in mb]
        raw_eigenvectorsqs = [sample['eigenvectors_sq'] for sample in mb]
        raw_complexes = [sample['simplices'] for sample in mb]

        reindex_slices = np.cumsum([0] + [len(s) for s in raw_eigenvalues])

        flat_list_eigs = torch.tensor([item for sublist in raw_eigenvalues for item in sublist], dtype=torch.float).reshape(-1, 1)

        evs_transformed = torch.zeros([self.filters, len(flat_list_eigs)])

        x = torch.zeros([L * self.final_vec_length])
        bottom = 0
        for j in range(self.filters):
            evs_transformed = self.eigfs[j](flat_list_eigs).squeeze(-1)
            fs = [torch.matmul(evs_transformed[reindex_slices[i]: reindex_slices[i+1]], raw_eigenvectorsqs[i]) for i in range(L)]
            fdetached = [f.detach().numpy() for f in fs]
            data = zip(fdetached, raw_complexes)
            pool = multiprocessing.Pool()
            masks = pool.starmap(EPH_fast2, data)
            pool.close()

            for i in range(L):
                Dgm0 = fs[i][masks[i][0]]
                Dgm1 = fs[i][masks[i][1]]

                path0 = torch.sort(self.projections0[j](Dgm0.unsqueeze(0).unsqueeze(0)).squeeze(0), dim = -1).values
                top = bottom + self.siglength[0]
                x[bottom : top] += SigFn.apply(path0.transpose(0,1), self.pslevel[0]).squeeze(0)
                bottom = top

                path1 = torch.sort(self.projections1[j](Dgm1.unsqueeze(0).unsqueeze(0)).squeeze(0), dim = -1).values
                top = bottom + self.siglength[1]
                x[bottom : top] += SigFn.apply(path1.transpose(0,1), self.pslevel[1]).squeeze(0)
                bottom = top

        x = x.reshape(self.filters, L, sum(self.siglength)).transpose(0,1)
        x = x.reshape(L, self.final_vec_lengthm)
        x = self.final_layer(x)

        return x


class dgmslice_ps(nn.Module):

    def __init__(self, ef_num_hlayers, ef_num_hu, ef_num_filters, pslevel, num_slices, dropout):

        super(dgmslice_ps, self).__init__()

        self.filters = ef_num_filters
        self.pslevel = pslevel
        self.siglength = [iisignature.siglength(num_slices+1, m) for m in pslevel]
        self.eigfs = nn.ModuleList()
        self.final_vec_length = self.filters * sum(self.siglength)
        self.slices = num_slices
        for i in range(self.filters):
            self.eigfs.append(eigf(ef_num_hlayers, ef_num_hu))

        self.projections = nn.Conv1d(1,self.slices,  2, 2)
        self.final_layer = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.final_vec_length, 1), nn.BatchNorm1d(1))

    def forward(self, mb):

        L = len(mb)
        raw_eigenvalues = [sample['eigenvalues'] for sample in mb]
        raw_eigenvectorsqs = [sample['eigenvectors_sq'] for sample in mb]
        raw_complexes = [sample['simplices'] for sample in mb]

        reindex_slices = np.cumsum([0] + [len(s) for s in raw_eigenvalues])

        flat_list_eigs = torch.tensor([item for sublist in raw_eigenvalues for item in sublist], dtype=torch.float).reshape(-1, 1)

        evs_transformed = torch.zeros([self.filters, len(flat_list_eigs)])

        x = torch.zeros([L, self.final_vec_length])

        for j in range(self.filters):
            evs_transformed[j, :] = self.eigfs[j](flat_list_eigs).squeeze(-1)

        for i in range(L):
            f = torch.matmul(evs_transformed[:, reindex_slices[i]: reindex_slices[i+1]], raw_eigenvectorsqs[i])
            bottom = 0
            for j in range(self.filters):

                dgm0, dgm1ext, dgm1rel = EPH_fast2(f[j].detach().numpy(), raw_complexes[i]) #output contiguous (b,d, b, d)

                #Dgm0 = torch.cat((f[j][dgm0], f[j][dgm1rel])) #H1rel on same side of the diagonal
                Dgm0 = torch.cat((f[j][dgm0], f[j][dgm1rel][torch.arange(max(len(dgm1rel)-1,0), 0, -1)])) #H1rel on opposite sides
                pathlength = int(len(Dgm0)/2)
                path0 = torch.zeros([self.slices + 1, pathlength], dtype=torch.float)
                path0[0] = torch.arange(pathlength)/(pathlength*(pathlength+1)/2)
                path0[1:] = torch.sort(self.projections(Dgm0.unsqueeze(0).unsqueeze(0)).squeeze(0), dim = -1).values
                top = bottom + self.siglength[0]
                x[i, bottom : top] += SigFn.apply(path0.transpose(0,1), self.pslevel[0]).squeeze(0)
                bottom = top

                Dgm1 = f[j][dgm1ext]
                pathlength = int(len(Dgm1)/2)
                path1 = torch.zeros([self.slices + 1, pathlength], dtype=torch.float)
                path1[0] = torch.arange(pathlength)/(pathlength*(pathlength+1)/2)
                path1[1:] = torch.sort(self.projections(Dgm1.unsqueeze(0).unsqueeze(0)).squeeze(0), dim = -1).values

                top = bottom + self.siglength[1]
                x[i, bottom : top] += SigFn.apply(path1.transpose(0,1), self.pslevel[1]).squeeze(0)
                bottom = top

        x = self.final_layer(x)

        return x
