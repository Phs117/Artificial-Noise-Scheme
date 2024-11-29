# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:23:16 2023

@author: Ândrei Camponogara
"""

import numpy as np


class OFDM:

    def __init__(self, N, Ncp, map):
        self.map = map                   # map = 1 -> DMT / map = 0 -> OFDM
        self.N = N + N*self.map          # Comprimento do bloco de símbolo
        self.Ncp = Ncp                   # Comprimento do prefíxo cíclico
        
        if (map != 1) or (map != 0): 
            raise ValueError('map deve ser igual a 1 ou 0')
            
    def mod(self, X):
        """
        
        Realiza a modulação DMT e OFDM.
    
        Entrada
        ----------
        X : Matriz com os símbolos provinientes de modulação digital M-ária cujas
            linhas representa as subportadoras e as colunas os blocos de símbolo
            DMT/OFDM
    
        Saída
        -------
        x_n : Sequência.
    
        """
    
        # DMT
        if self.map == 1:
            # Mapeamento
            X_map = np.vstack((np.real(X[-1]), X[:-1], np.imag(X[-1]), np.conj(X[-2::-1])))
            
        # OFDM
        else:
            X_map = X
            
        # IFFT - símbolos OFDM
        x = np.real(np.fft.ifft(X_map, axis = 0, norm = "ortho"))
    
        # Inserção do prefixo cíclico
        x_cp =  np.vstack((x[self.N - self.Ncp :], x))
    
        # Paralelo para serial
        x_n = x_cp.reshape(-1, order = 'F')
        
        return x_n
    
    def demod(self, y_n):
        """
        Realiza a demodulação DMT e OFDM.
    
        Entrada
        ----------
        y_n : Vetor de bloco de símbolos DMT/OFDM concatenados.
    
        Saída
        -------
        Y_map : Matriz com os blocos de símbolos no domínio da frequência cujas
                colunas representam os blocos de símbolo DMT/OFDM e as linhas as 
                subportadoras.
    
        """
        
        # Número de blocos de símbolo DMT/OFDM
        Ns = int(len(y_n)/(self.N + self.Ncp))
        
        # Serial para paralelo
        y_cp = y_n.reshape(self.N + self.Ncp, Ns, order = 'F')
        
        # Remoção do prefixo cíclico
        y =  y_cp[self.Ncp:]
        
        # Transformada de Fourier
        Y_map = np.fft.fft(y, axis = 0, norm = "ortho") 
    
        return Y_map