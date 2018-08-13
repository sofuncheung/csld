# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 20:00:49 2018

@author: sofuncheung
"""

import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg
from sympy import Matrix
from numpy.linalg import svd
import os
import copy
import itertools


try:
    import cStringIO as StringIO
except ImportError:
    from io import StringIO
try:
    import hashlib
    hashes=True
except ImportError:
    hashes=False

def read_POSCAR():
    """
    Return all the relevant information contained in a POSCAR file.
    """
    nruter=dict()
    nruter["lattvec"]=np.empty((3,3))
    f=open("POSCAR","r")
    firstline=f.__next__()
    factor=float(f.__next__().strip())
    for i in range(3):
        nruter["lattvec"][:,i]=[float(j) for j in f.__next__().split()]
    nruter["lattvec"]*=factor
    line=f.__next__()
    fields=f.__next__().split()
    old=False
    try:
        int(fields[0])
    except ValueError:
        old=True
    if old:
        nruter["elements"]=firstline.split()
        nruter["numbers"]=np.array([int(i) for i in line.split()])
        typeline="".join(fields)
    else:
        nruter["elements"]=line.split()
        nruter["numbers"]=np.array([int(i) for i in fields],
                                   dtype=np.intc)
        typeline=f.__next__()
    natoms=nruter["numbers"].sum()
    nruter["positions"]=np.empty((3,natoms))
    for i in range(natoms):
        nruter["positions"][:,i]=[float(j) for j in f.__next__().split()]
    f.close()
    nruter["types"]=[]
    for i in range(len(nruter["numbers"])):
        nruter["types"]+=[i]*nruter["numbers"][i]
    if typeline[0]=="C":
        nruter["positions"]=sp.linalg.solve(nruter["lattvec"],
                                               nruter["positions"]*factor)
    return nruter

def write_POSCAR(poscar,filename):
    """
    Write the contents of poscar to filename.
    """
    global hashes
    f=StringIO()
    f.write("1.0\n")
    for i in range(3):
        f.write("{0[0]:>20.15f} {0[1]:>20.15f} {0[2]:>20.15f}\n".format(
            (poscar["lattvec"][:,i]).tolist()))
    f.write("{0}\n".format(" ".join(poscar["elements"])))
    f.write("{0}\n".format(" ".join([str(i) for i in poscar["numbers"]])))
    f.write("Direct\n")
    for i in range(poscar["positions"].shape[1]):
        f.write("{0[0]:>20.15f} {0[1]:>20.15f} {0[2]:>20.15f}\n".format(
            poscar["positions"][:,i].tolist()))
    if hashes:
        header=hashlib.sha1(f.getvalue().encode()).hexdigest()
    else:
        header=filename
    with open(filename,"w") as finalf:
        finalf.write("{0}\n".format(header))
        finalf.write(f.getvalue())
    f.close()

def random_sign():
    if np.random.randint(0,2) == 0: return 1
    else: return -1

def move_atoms(poscar):
    """
    Return a copy of poscar with each atom iat displaced by random \Aring along
    its icoord-th Cartesian coordinate. 
    """
    nruter=copy.deepcopy(poscar)
    displist = np.array([0.,0,0])
    ntot = nruter["positions"].shape[1]
    for iat in range(ntot):
        disp = (0.01 * np.random.randn(3) + 0.03) * random_sign()
        displist = np.vstack((displist,disp)) # will the vstack here drag the speed?
        nruter["positions"][:,iat]+=scipy.linalg.solve(nruter["lattvec"],
                                                       disp)
    nruter['displist'] = displist[1:]  
    #nruter['u_0'] = np.max(displist[1:])
    return (nruter)





def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def null(a, eps=1e-3):
    u, s, vh = scipy.linalg.svd(a)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0) #挑选出奇异值为 0 的行
    return scipy.transpose(null_space)



def kron(r, times):
    if times < 2: 
        raise ValueError
    elif times == 2:
        return np.kron(r, r)
    else:
        temp = np.kron(r, r)
        for i in range(times-2):
            temp = np.kron(temp, r)
        return temp
        
def gauss_jordan(m, eps = 1e-10):
    '''
    Puts given matrix (2D array) into the Reduced Row Echelon Form.
    Returns True if successful, False if 'm' is singular.
    NOTE: make sure all the matrix items support fractions! Int matrix will NOT work!
    Written by J. Elonen in April 2005, released into Public Domain
    #### This algorithm can't handle single matrix. Does no good to the null
    #### space analyze.
    '''
    (h, w) = (len(m), len(m[0]))
    for y in range(0,h):
        maxrow = y
        for y2 in range(y+1, h):    # Find max pivot
            if abs(m[y2][y]) > abs(m[maxrow][y]):
                maxrow = y2
        (m[y], m[maxrow]) = (m[maxrow], m[y])
        if abs(m[y][y]) <= eps:     # Singular?
            return False
        for y2 in range(y+1, h):    # Eliminate column y
            c = m[y2][y] / m[y][y]
            for x in range(y, w):
                m[y2][x] -= m[y][x] * c
    for y in range(h-1, 0-1, -1): # Backsubstitute
        c = m[y][y]
        for y2 in range(0,y):
            for x in range(w-1, y-1, -1):
                m[y2][x] -=  m[y][x] * m[y2][y] / c
        m[y][y] /= c
        for x in range(h, w):       # Normalize row y
            m[y][x] /= c
    return True



def cluster_factorial(cluster):
    '''
    Return the 'factorial' of a cluster. Both proper and improper. 
    '''
    index = cluster.index
    dic = {}
    factorial = 1
    for i in index:
        if index.count(i) > 0:
            dic[i] = index.count(i)
    for value in dic.values():
        factorial = factorial * np.math.factorial(value)
    return factorial
            
def cluster_identical(cluster_a, cluster_b):
    '''
    Return whether two clusters are identical.
    '''
    a = copy.deepcopy(cluster_a.index)
    b = copy.deepcopy(cluster_b.index)
    a.sort()
    b.sort()
    return a == b

def displacements_product(tuple_list, main_tuple, poscar):
    tuple_list_new = copy.deepcopy(tuple_list)
    tuple_list_new.remove(main_tuple)
    product = 1.
    for j in tuple_list_new:
        if j[1] == 'x':    
            product = product * poscar['displist'][j[0]][0]
        if j[1] == 'y':
            product = product * poscar['displist'][j[0]][1]
        if j[1] == 'z':
            product = product * poscar['displist'][j[0]][2]
    return product

def generating_sensing_mtx(N_atoms, clusters, poscar):
    A = -1 * np.ones([3 * N_atoms, 1])
    for orbit in clusters:
        sensing_mtx_block = np.zeros([3 * N_atoms, np.power(3, orbit[0].size)])
        temp_block = np.zeros([3 * N_atoms, np.power(3, orbit[0].size)])
        for cluster in orbit:
            is_rep = cluster.is_rep
            if not(is_rep):
                #for i in sorted(set(cluster.index),key=cluster.index.index) :# repeat atoms only need to be considered once.
                #    order = cluster.index.count(i)
                #    temp_block[3 * i, ] = -1./cluster_factorial(cluster)*
                for index, item in enumerate(itertools.product(['x', 'y', 'z'], repeat=orbit[0].size)):
                    tuple_list = []
                    for i in range(len(item)):
                        tuple_list.append((cluster.index[i], item[i]))
                    for j in sorted(set(tuple_list),key=tuple_list.index):
                        order = tuple_list.count(j)
                        if j[1] == 'x':
                            temp_block[3*j[0], index] = order * displacements_product(tuple_list, j, poscar)
                        if j[1] == 'y':
                            temp_block[3*j[0]+1, index] = order * displacements_product(tuple_list, j, poscar)
                        if j[1] == 'z':
                            temp_block[3*j[0]+2, index] = order * displacements_product(tuple_list, j, poscar)
                temp_block = np.dot(temp_block, kron(cluster.point, orbit[0].size))* (-1. / cluster_factorial(cluster))
                sensing_mtx_block += temp_block
        A = np.hstack((A, sensing_mtx_block))
    return A

def normalize_sensing_mtx(A, clusters):
    B = copy.deepcopy(A)
    temp = [0]
    for i, orbit in enumerate(clusters):
        if orbit[0].size > clusters[temp[-1]][0].size:
            temp.append(i)
        else:
            continue
    #highest_order = len(temp) + 1
    max_list = []
    for i in range(len(temp)):
        if i == 0:
            start_column = 1
            end_column = (temp[i+1]-temp[i])*3**(i+2)+1
               
        if i == len(temp) - 1:
            start_column = 1
            for j in range(len(temp[:i])):
                start_column += (temp[j+1]-temp[j])*3**(j+2)
            end_column = B.shape[1]
        else:
            start_column = 1
            for j in range(len(temp[:i])):
                start_column += (temp[j+1]-temp[j])*3**(j+2)
            end_column = start_column + (temp[i+1]-temp[i])*3**(i+2)
            
        #print(start_column, end_column)
        u_0 = np.max(abs(B[:, start_column:end_column]))
        B[:, start_column:end_column] = B[:, start_column:end_column] / u_0
        max_list.append(u_0)
        
    return B, max_list
    

    
def componont_equal_by_sym(a, b, index):
    '''
    a, b are (x, x, y, y, z, z) stuff produced by itertool.product('x','y','z')
    '''
    dic = {}
    for i in index:
        temp = [x for x in range(len(index)) if index[x] == i]
        dic[i] = temp
    for key in dic:
        temp_a = []
        temp_b = []
        for i in dic[key]:
            temp_a.append(a[i])
            temp_b.append(b[i])
        if temp_a.count('x') == temp_b.count('x') and temp_a.count('y') == temp_b.count('y') and temp_a.count('z') == temp_b.count('z'):
            continue
        else:
            return False
    return True
        
    

def generating_improper_cluster_constraint(index):
    '''
    For an improper cluster, because of the commutativity as partial derivatives, 
    there will be a constraint matrix satisfying B \Phi = 0.
    This function find the B for each improper cluster.index
    '''
    size = len(index)
    rows = np.power(3, size)
    #B = np.zeros([rows, columns_of_A])
    constraint_block = np.zeros([rows, rows])  
    temp = []
    for i,j in enumerate(itertools.product(['x', 'y', 'z'], repeat=size)):
        temp.append(j)
    for i in range(rows):
        for j in range(i+1, rows):
            if componont_equal_by_sym(temp[i], temp[j], index):
                constraint_block[i][i] = 1
                constraint_block[i][j] = -1
                break
    return constraint_block
    
def position_in_clusters(cluster, clusters):
    
    
    
    return position


def generating_further_reducing_mtx(clusters):
    eps = 1e-8
    nb_of_compononts = 0
    for orbit in clusters:
        nb_of_compononts += np.power(3, orbit[0].size)
    #C = np.eye()
    for orbit in clusters:
        B_list = []
        if orbit[0].proper == False:
            block = generating_improper_cluster_constraint(orbit[0].index)
            B = np.zeros([np.power(3, orbit[0].size), nb_of_compononts+1])
            B[:, position_in_clusters(orbit[0], clusters):position_in_clusters(orbit[0], clusters)+np.power(3, orbit[0].size)] = block
            B_list.append(B)
            



    
def clusters_from_file(filename):
    '''
    from the 'log' generate a list containing each representative cluster and its orbit clusters. 
    '''
    lat_in = cluster_file_indexing('lat.in')
    clusters = []
    block_start_list = []
    cluster_blocks = []
    with open(filename, 'r') as f:
        content = f.readlines()
        for index, line in enumerate(content):
            if 'Representative cluster:' in line and int(content[index+1]) > 1:
                block_start_list.append(index)
        for i, block_start in enumerate(block_start_list):
            if i != len(block_start_list) - 1:
                cluster_blocks.append(content[block_start_list[i]: block_start_list[i+1]])
            else:
                cluster_blocks.append(content[block_start_list[i]:])
                
                
        for index, block in enumerate(cluster_blocks):
            cluster_size = int(block[1])
            orbit = []
            orbit.append(representative_cluster(block[2: 2+cluster_size], [lat_in.point2index(i, lat_in.atoms_list) for i in block[2: 2+cluster_size]])) 
            for i, line in enumerate(block):
                #current_cluster = []
                #for i in range(cluster_order):
                    #current_cluster.append(lat_in.point2index(content[index+i+2], lat_in.atoms_list))
                #break
                if 'Symmetry operation' in line and 'Point' in block[i+1] and 'Translation' in block[i+6] and 'Transformed cluster' in block[i+9]:
                    orbit.append(hidden_cluster(block[i+2:i+5], block[i+7], block[2: 2+cluster_size], block[i+11: i+11+cluster_size], [lat_in.point2index(i, lat_in.atoms_list) for i in block[i+11: i+11+cluster_size]]))
            clusters.append(orbit)
    return clusters
                
                
                

class representative_cluster:
    def __init__(self, cluster, index):
        self.size = len(cluster)
        self.atoms = cluster
        self.is_rep = True
        self.hidden = False
        self.index = index
        self.proper = len(self.index) == len(set(self.index))
       
    
class hidden_cluster:
    '''
    Those clusters can be transformed from representative clusters by symmetry operations.
    '''
    def __init__(self, point, trans, rep_cluster, cluster, index):
        self.size = len(cluster)
        self.point = self.point_to_nparray(point)
        self.trans = trans
        self.rep = rep_cluster
        self.atoms = cluster
        self.is_rep = False
        self.hidden = True
        self.index = index
        self.proper = len(self.index) == len(set(self.index))
            
    def point_to_nparray(self, point):
        nparray = np.zeros([3, 3])
        for i, j in enumerate(point):
            nparray[i, 0] = float(j.split()[0])
            nparray[i, 1] = float(j.split()[1])
            nparray[i, 2] = float(j.split()[2])
        return nparray
        
class cluster_file_indexing:
    '''
    deal with the 'lat.in', get lattice parameters and index of each atom.
    '''
    def __init__(self, filename):
        with open(filename, 'r') as f:
            content = f.readlines()
            first_line = content[0]
            if first_line.split()[-3]=='90' and first_line.split()[-2]=='90' and first_line.split()[-1]=='90':
                self.lattice_para_a = float(content[1].split()[0])
                self.lattice_para_b = float(content[2].split()[1])
                self.lattice_para_c = float(content[3].split()[2])
                self.atoms_list = content[4:]
            else:
                print('The lat.in file is not Orthogonal!!!')

        
    def alike(self, a, b):
        '''
        return whether 2 atoms are the same. input a and b are string.  
        '''
        a1 = float(a.split()[0])
        a2 = float(a.split()[1])
        a3 = float(a.split()[2])
        b1 = float(b.split()[0])
        b2 = float(b.split()[1])
        b3 = float(b.split()[2])
        if (abs(a1-b1)<1e-4 or abs(abs(a1-b1)-self.lattice_para_a)<1e-4) & (abs(a2-b2)<1e-4 or abs(abs(a2-b2)-self.lattice_para_b)<1e-4) & (abs(a3-b3)<1e-4 or abs(abs(a3-b3)-self.lattice_para_c)<1e-4):
            return True
        else:	
            return False 
        
    def point2index(self, string, atoms_list):
        for i, a in enumerate(atoms_list):
            if self.alike(a, string):
                return(i)
                break


class Sym_mtx_constrain:
    '''
    Mind this only suitable for {aa...a}.
    '''
    def __init__(self, order):
        self.order = order
        self.n = np.power(3, order)
    
    def equal_by_sym(self, a, b):
        '''
        a, b are index of FCTs. This func maps a, b into a 3-len list of how many 'x',
        'y', 'z' in 'a' or 'b'. if two list are identical then a and b are equal_by_sym.
        '''
        ''' testing
        f1 = a // (self.n / 3) # 0 -> x; 1 -> y; 2 -> z.
        f2 = (a % (self.n / 3)) // (self.n / 9)
        f3 = (a % (self.n / 3)) % (self.n / 9)
        '''
        F1 = [] # feature
        F2 = []
        for i in range(self.order - 1):
            f1 = a // (np.power(3, self.order - i - 1))
            a = a % (np.power(3, self.order - i - 1))
            F1.append(f1)
            f2 = b // (np.power(3, self.order - i - 1))
            b = b % (np.power(3, self.order - i - 1))
            F2.append(f2)
        F1.append(a)
        F2.append(b)
        if F1.count(0) == F2.count(0) and F1.count(1) == F2.count(1) and F1.count(2) == F2.count(2):
            return True
        else:
            return False
              
    
    def generate(self):
        '''
        Generate the linear constraint matrix describing the symmertric matrix.
        '''
        A = np.zeros((self.n, self.n))
        for row in range(self.n):
            for col in range(row + 1, self.n):
                if self.equal_by_sym(row, col):
                    A[row][row] = 1
                    A[row][col] = -1
                    break
        return A
            



eps = 1e-8

'''这个是ATAT产生的NaCl'''
'''
r = np.zeros([6,3,3])

r[0] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

r[1] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

r[2] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

r[3] = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

r[4] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

r[5] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

G = []
for i in range(6):
    temp = np.kron(r[i], r[i])
    n = temp.shape[0]
    eye = np.eye(n)
    G.append(temp - eye)
'''   
 

order = 2
   
'''这个是晶体对称性网站上 m -3 m的generators'''

'''
r_1 = np.zeros([5,3,3])
r_1[0] = np.array([[-1, 0, 0], 
                   [0, -1, 0], 
                   [0, 0, 1]])
    
r_1[1] = np.array([[-1, 0, 0], 
                   [0, 1, 0], 
                   [0, 0, -1]])

r_1[2] = np.array([[0, 0, 1], 
                   [1, 0, 0], 
                   [0, 1, 0]])

r_1[3] = np.array([[0, 1, 0], 
                   [1, 0, 0], 
                   [0, 0, -1]])
    
r_1[4] = np.array([[-1, 0, 0], 
                   [0, -1, 0], 
                   [0, 0, -1]])
'''

'''这些是 ATAT 产生的关于NaCl的迷向子群对应的旋转矩阵'''
r_1 = np.zeros([8,3,3])
r_1[0] = np.array([[1, 0, 0], 
                   [0, 0, 1], 
                   [0, 1, 0]])
    
r_1[1] = np.array([[1, 0, 0], 
                   [0, 1, 0], 
                   [0, 0, 1]])

r_1[2] = np.array([[-1, 0, 0], 
                   [0, 0, 1], 
                   [0, 1, 0]])

r_1[3] = np.array([[0, -1, 0], 
                   [0, 0, 1], 
                   [-1, 0, 0]])
    
r_1[4] = np.array([[0, -1, 0], 
                   [-1, 0, 0], 
                   [0, 0, 1]]) 
    
r_1[5] = np.array([[-1, 0, 0], 
                   [0, 1, 0], 
                   [0, 0, 1]])   
    
r_1[6] = np.array([[0, 0, -1], 
                   [-1, 0, 0], 
                   [0, 1, 0]])

r_1[7] = np.array([[0, 0, -1], 
                   [0, 1, 0], 
                   [-1, 0, 0]])     
    

       
G1 = []
n = np.power(3, order)
C1 = np.eye(n)
for i in range(np.shape(r_1)[0]):
    temp = kron(r_1[i], order)
    #n = temp.shape[0]
    eye = np.eye(n)
    B = temp - eye
    G1.append(B)

S = Sym_mtx_constrain(order)
G1.append(S.generate())    
for i in range(np.shape(r_1)[0] + 1):
    B_C = Matrix(np.dot(G1[i], C1))
    
    
    C_prim = B_C.nullspace()
    C_prim = np.array(C_prim).T
    C_prim = C_prim.astype(np.float64)   
    C_prim = np.where(abs(C_prim) < eps, 0, C_prim)
    if np.shape(C_prim) == (0,):
        print('This kind of FCTs do not exist!!!')
        break
    else:    
        C1 = np.dot(C1, C_prim)
   


'''这个是晶体对称性网站上 4 -3 m的generators'''


r_1 = np.zeros([4,3,3])
r_1[0] = np.array([[-1, 0, 0], 
                   [0, -1, 0], 
                   [0, 0, 1]])
    
r_1[1] = np.array([[-1, 0, 0], 
                   [0, 1, 0], 
                   [0, 0, -1]])

r_1[2] = np.array([[0, 0, 1], 
                   [1, 0, 0], 
                   [0, 1, 0]])

r_1[3] = np.array([[0, 1, 0], 
                   [1, 0, 0], 
                   [0, 0, 1]])
    

G2 = []

#see = []
#cprim = []

n = np.power(3, order)
C2 = np.eye(n)
for i in range(np.shape(r_1)[0]):
    temp = kron(r_1[i], order)
    #n = temp.shape[0]
    eye = np.eye(n)
    B = temp - eye
    G2.append(B)
 
S = Sym_mtx_constrain(order)
G2.append(S.generate())        
for i in range(np.shape(r_1)[0] + 1):    
    B_C = Matrix(np.dot(G2[i], C2))
    C_prim = B_C.nullspace()
    C_prim = np.array(C_prim).T
    C_prim = C_prim.astype(np.float64)   
    
    #cprim.append([C_prim])
    
    C_prim = np.where(abs(C_prim) < eps, 0, C_prim)
    
    #see.append([C2])
    if np.shape(C_prim) == (0,):
        print('This kind of FCTs do not exist!!!')
        break
    else:    
        C2 = np.dot(C2, C_prim)
    

'''
l = []
for i in range(3):
    if i==0: t1 ='x'
    elif i==1: t1='y'
    elif i==2: t1='z'
    for j in range(3):
        if j==0: t2 ='x'
        elif j==1: t2='y'
        elif j==2: t2='z'
        for k in range(3):
            if k==0: t3 ='x'
            elif k==1: t3='y'
            elif k==2: t3='z'
            for m in range(3):
                if m==0: t4 ='x'
                elif m==1: t4='y'
                elif m==2: t4='z'
                str0 = t1+t2+t3+t4
                l.append(str0)
'''

     