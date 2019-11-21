import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import dionysus
import pickle
import numpy as np
import multiprocessing
import time

def EPH_mask(vertex_values, simplices):

    s2v_lst = [[sorted(s), sorted([[vertex_values[v],v] for v in s], key=lambda x: x[0])] for s in simplices]
    f_ord = [dionysus.Simplex(s[0], s[1][-1][0]) for s in s2v_lst] #takes max
    f_ext = [dionysus.Simplex([-1] + s[0], s[1][0][0]) for s in s2v_lst] #takes min

    ord_dict = {tuple(s[0]): s[1][-1][1] for s in s2v_lst}
    ext_dict = {tuple([-1] + s[0]): s[1][0][1] for s in s2v_lst}

    f_ord.sort(key = lambda s: (s.data, len(s)))
    f_ext.sort(key = lambda s: (-s.data, len(s)))

    #computes persistence
    f = dionysus.Filtration([dionysus.Simplex([-1], -float('inf'))] + f_ord + f_ext)
    m = dionysus.homology_persistence(f)

    dgms = [[[], []], [[], []], [[], []], [[], []]] #H0ord, H0ext, H1rel, H1ext

    for i in range(len(m)):

        dim = f[i].dimension()

        if m.pair(i) < i: continue      # skip negative simplices to avoid double counting
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
                    if dim == 0: #H0ext
                        dgms[0][0].append(ord_dict[tuple(pos)])
                        dgms[0][1].append(ext_dict[tuple(neg)])

                    if dim == 1: #H1ext
                        dgms[3][0].append(ext_dict[tuple(neg)])
                        dgms[3][1].append(ord_dict[tuple(pos)])

    return dgms


def EPH_fast3(vertex_values, simplices):

    N =len(vertex_values)
    order2vertex = np.argsort(vertex_values)
    vertex2order = np.empty(N, dtype=int)
    vertex2order[order2vertex] = np.arange(N)

    f_ord = [(s, max(vertex2order[v] for v in s)) for s in simplices] #takes max
    f_ext = [([-1] + s, min(vertex2order[v] for v in s)) for s in simplices] #takes min

    f_ord.sort(key = lambda s: (s[1], len(s[0])))
    f_ext.sort(key = lambda s: (-s[1], len(s[0])))

    full_filtration = [ ([-1], -float('inf'))] + f_ord + f_ext

    f = dionysus.Filtration(full_filtration)
    m = dionysus.homology_persistence(f)

    dgms = dionysus.init_diagrams(m, f)

    H0 = list(dgms[0])
    H0.remove(max(H0, key=lambda x: x.death))
    Dgm0, Dgm1 = [[], []], [[], []]

    for pt in H0:

        Dgm0[0].append(order2vertex[int(pt.birth)])
        Dgm0[1].append(order2vertex[int(pt.death)])

    H1 = list(dgms[1])
    index_cutoff = len(f_ord) +1

    for pt in H1:
        if pt.data < index_cutoff:
            Dgm1[0].append(order2vertex[int(pt.death)])
            Dgm1[1].append(order2vertex[int(pt.birth)])
        else:
            Dgm0[0].append(order2vertex[int(pt.death)])
            Dgm0[1].append(order2vertex[int(pt.birth)])

    return Dgm0, Dgm1



def EPH_fast2(vertex_values, simplices):

    N =len(vertex_values)
    order2vertex = np.argsort(vertex_values)
    vertex2order = np.empty(N, dtype=int)
    vertex2order[order2vertex] = np.arange(N)

    f_ord = [(s, max(vertex2order[v] for v in s)) for s in simplices] #takes max
    f_ext = [([-1] + s, min(vertex2order[v] for v in s)) for s in simplices] #takes min

    f_ord.sort(key = lambda s: (s[1], len(s[0])))
    f_ext.sort(key = lambda s: (-s[1], len(s[0])))

    full_filtration = [ ([-1], -float('inf'))] + f_ord + f_ext

    f = dionysus.Filtration(full_filtration)
    m = dionysus.homology_persistence(f)

    dgms = dionysus.init_diagrams(m, f)

    H0 = list(dgms[0])
    H0.remove(max(H0, key=lambda x: x.death))
    Dgm0, Dgm1 = [], []

    for pt in H0: Dgm0 += [order2vertex[int(pt.birth)], order2vertex[int(pt.death)]]

    H1 = list(dgms[1])
    index_cutoff = len(f_ord) +1

    for pt in H1:
        if pt.data < index_cutoff:
            Dgm1 += [order2vertex[int(pt.birth)], order2vertex[int(pt.death)]]
        else:
            Dgm0 += [order2vertex[int(pt.death)], order2vertex[int(pt.birth)]]

    return Dgm0, Dgm1


def EPH_fast(vertex_values, simplices):

    N =len(vertex_values)
    order2vertex = np.argsort(vertex_values)
    vertex2order = np.empty(N, dtype=int)
    vertex2order[order2vertex] = np.arange(N)

    f_ord = [(s, max(vertex2order[v] for v in s)) for s in simplices] #takes max
    f_ext = [([-1] + s, min(vertex2order[v] for v in s)) for s in simplices] #takes min

    f_ord.sort(key = lambda s: (s[1], len(s[0])))
    f_ext.sort(key = lambda s: (-s[1], len(s[0])))

    full_filtration = [ ([-1], -float('inf'))] + f_ord + f_ext

    f = dionysus.Filtration(full_filtration)
    m = dionysus.homology_persistence(f)

    dgms = dionysus.init_diagrams(m, f)

    H0 = list(dgms[0])
    H0.remove(max(H0, key=lambda x: x.death))
    Dgm0b = [order2vertex[int(pt.birth)] for pt in H0]
    Dgm0d = [order2vertex[int(pt.death)] for pt in H0]

    H1 = list(dgms[1])
    index_cutoff = len(f_ord) +1

    Dgm1extb = [order2vertex[int(pt.birth)] for pt in H1 if pt.data < index_cutoff]
    Dgm1extd = [order2vertex[int(pt.death)] for pt in H1 if pt.data < index_cutoff]

    Dgm0b += [order2vertex[int(pt.birth)] for pt in H1 if pt.data >= index_cutoff]
    Dgm0d += [order2vertex[int(pt.death)] for pt in H1 if pt.data >= index_cutoff]

    return [[Dgm0b, Dgm0d], [Dgm1extd, Dgm1extb]]


def EPH_mask_demo(vertex_values, simplices):
    times = []
    t0 = time.time()

    s2v_lst = [[sorted(s), sorted([[vertex_values[v],v] for v in s], key=lambda x: x[0])] for s in simplices]
    f_ord = [dionysus.Simplex(s[0], s[1][-1][0]) for s in s2v_lst] #takes max
    f_ext = [dionysus.Simplex([-1] + s[0], s[1][0][0]) for s in s2v_lst] #takes min

    ord_dict = {tuple(s[0]): s[1][-1][1] for s in s2v_lst}
    ext_dict = {tuple([-1] + s[0]): s[1][0][1] for s in s2v_lst}

    t1 = time.time()
    times.append(t1 -t0)
    t0 = time.time()

    f_ord.sort(key = lambda s: (s.data, len(s)))
    f_ext.sort(key = lambda s: (-s.data, len(s)))

    t1 = time.time()
    times.append(t1 -t0)
    t0 = time.time()
    #computes persistence
    f = dionysus.Filtration([dionysus.Simplex([-1], -float('inf'))] + f_ord + f_ext)
    m = dionysus.homology_persistence(f)
    t1 = time.time()
    times.append(t1 -t0)
    t0 = time.time()


    dgms = [[[], []], [[], []], [[], []], [[], []]] #H0ord, H0ext, H1rel, H1ext

    for i in range(len(m)):

        dim = f[i].dimension()

        if m.pair(i) < i: continue      # skip negative simplices to avoid double counting
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
                    if dim == 0: #H0ext
                        dgms[0][0].append(ord_dict[tuple(pos)])
                        dgms[0][1].append(ext_dict[tuple(neg)])

                    if dim == 1: #H1ext
                        dgms[3][0].append(ext_dict[tuple(neg)])
                        dgms[3][1].append(ord_dict[tuple(pos)])

    t1 = time.time()
    times.append(t1 -t0)
    t0 = time.time()

    dgms = dionysus.init_diagrams(m, f)

    t1 = time.time()
    times.append(t1 -t0)
    return times
