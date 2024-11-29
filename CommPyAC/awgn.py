# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 11:40:15 2023

@author: Ândrei Camponogara
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


#%% Probabilidade de erro
def Q(x):
    """
    Função Q.

    """
    return 0.5 - 0.5*erf(x/np.sqrt(2))


class Pe():
    '''
        Probabilidade de erro
    '''
    
    def __init__(self, M, EbN0):
        self.M = M          # Ordem da constelação
        self.b = np.log2(M) # Bits por símbolo
        self.EbN0 = EbN0

    def pam(self):
        '''
        Calcula a probabilidade de erro da modulação M-PSK em canal AWGN.
        
        '''
    
        Pe = 2*(1-1/self.M)*Q(np.sqrt(6/(self.M**2-1)*self.b*self.EbN0))
        
        return Pe


    def psk(self):
        '''
        Calcula a probabilidade de erro da modulação M-PSK em canal AWGN.
        
        '''
        
        if self.M > 4:
            Pe = 2*Q(np.sqrt(2*self.b*self.EbN0)*np.sin(np.pi/self.M))
        elif self.M == 4:
            Pe = 2*Q(np.sqrt(self.b*self.EbN0)) - Q(np.sqrt(self.b*self.EbN0))**2
        else:
            Pe = Q(np.sqrt(2*self.b*self.EbN0))

        return Pe    


    def qam(self):
        """
        Calcula a probabilidade de erro da modulação M-QAM em canal AWGN.

        """
        
        Pe = 4*(1 - 1/np.sqrt(self.M))*Q(np.sqrt(3/(self.M-1)*self.b*self.EbN0))
        
        return Pe


#%% Probabilidade de erro média - OFDM

def ofdmPe(snr, h, N, M):
    H = np.fft.fft(h, 2*N, axis=0)
    snrs = (10**(snr/10)) * np.abs(H)**2
     
    EbN0 = snrs / np.log2(M)
    pe = Pe(M, EbN0)
    ser = pe.qam()
    
    return np.mean(ser)

#%% Ruído AWGN
def awgn(x, snr_dB):
    """
    Adiciona o ruído AWGN ao sinal s considerando a relação sinal ruído
    snr_dB

    Entradas
    ----------
    x : Sinal a ser afetado pelo ruído AWGN.
    snr_dB : Relação sinal-ruído em dB.

    Saída
    -------
    y : Sinal x afetado pelo ruído AWGN, respeitada a snr_dB.

    """
    
    # Comprimento do vetor x
    Nx = len(x)
    # SNR linear
    snr = 10 ** (snr_dB/10)
    # Potência de x
    Px = np.sum(abs(x) ** 2)/Nx
    # Potência do ruído (no caso de média zero)    
    sigma2 = Px/snr    
    # Geração do ruído AWGN
    if np.isrealobj(x):
        w = np.sqrt(sigma2) * np.random.randn(Nx)
    else: # CAWGN
        w = np.sqrt(sigma2/2) * (np.random.randn(Nx) + 1j*np.random.randn(Nx))
        
    # Sinal resultante da adição do ruído gerado
    y = x + w
    
    return y


