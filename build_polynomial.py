# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def build_poly(x, degree):
    ones_col = np.ones((len(x), 1))
    poly = x
    m, n = x.shape
    for deg in range(2, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    multi_indices = {}
    cpt = 0
    for i in range (n):
        for j in range(i+1,n):
            multi_indices[cpt] = [i,j]
            cpt = cpt+1
    
    gen_features = np.zeros(shape=(m, len(multi_indices)) )

    for i, c in multi_indices.items():
        gen_features[:, i] = np.multiply(x[:, c[0]],x[:, c[1]])

    poly =  np.c_[poly,gen_features]
    poly =  np.c_[ones_col,poly]

    return poly

#     res= np.empty((x.shape[0],(degree+1)*x.shape[1]))
#     for i in range(x.shape[0]):
#         for r in range(x.shape[1]):
#             for j in range(degree+1):
#                 res[i,r*(degree+1)+j]=x[i,r]**(j)           
#     return res



# def build_poly(x, **kwargs):
#     """polynomial basis function."""
#     degree = kwargs.get('degree')
#     new = np.array([ [ e**i for e in x ] for i in range(1,degree+1) ])
#     return new.T

# def build_poly_matrix(tx, degree):
#     """ apply build_poly to all columns of tx """
#     res = [ build_poly(x, degree=degree) for x in tx.T ]
#     conc = np.concatenate(res, axis=1)
#     one = np.ones((tx.shape[0], 1))
#     return np.concatenate([one, conc], axis=1)