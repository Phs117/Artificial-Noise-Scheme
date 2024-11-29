# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 21:00:14 2023

@author: acamp
"""

import numpy as np
from scipy.linalg import circulant, null_space


class Jamming:
    def __init__(self, N, Ncp, mapping):
        self.map = mapping               # map = 1 -> DMT / map = 0 -> OFDM
        self.N = N + N*self.map          # Comprimento do bloco de símbolo
        self.Ncp = Ncp  
        
        if (mapping != 1) and (mapping != 0): 
            raise ValueError('mapping deve ser igual a 1 ou 0')
    
    def an_cp(self, h, Ns, Ea):    
        # Matriz Rcp
        I = np.identity(self.N)
        O = np.zeros((self.N, self.Ncp))
        Rcp = np.hstack((O, I))
     
        
        # Matriz Rcp Ho
        hc = np.concatenate((h.ravel(), np.zeros((self.N + self.Ncp) - (len(h)))))
        H = circulant(hc)
        Ho = np.tril(H)
        RcpHo = Rcp@Ho
        
        # SVD --> espaço nulo de Haux
        Vn = null_space(RcpHo)
        
        # Geração do ruído artificial
        a = []
        for i in range(0, Ns):
            if self.map == 1:
                d = np.random.randn(self.Ncp)*np.sqrt(Ea)
            else:
                d = (np.random.randn(self.Ncp) + 1j*np.random.randn(self.Ncp))*np.sqrt(Ea)/2
            a_i = Vn@d # vetor AN 
            a = np.concatenate((a, a_i))
        return a 