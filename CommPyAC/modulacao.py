# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 10:32:39 2021

@author: Ândrei Camponogara
"""
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class PAM():
    '''
        M-PAM
    '''
    def __init__(self, M, Ex):
        self.M = M          # Constellation order
        self.Ex = Ex        # Average energy of the constellation
        # Minimum distance between two symbos with average energy Ex
        self.d = np.sqrt(12 * Ex / (self.M**2 - 1))
        
        if (M==1) or (np.mod(M,2)!=0): # M not a even power of 2
            raise ValueError('M must be even')

    def mod(self):
        '''
        Generate M-PAM symbols.
        
        Returns
        -------
        x : M-PAM symbols.

        '''
        
        # M-PAM constellation
        self.x = np.arange(-(self.M-1), self.M, 2) * self.d/2

        return self.x 
    
    def demod(self, y_k):
        '''
        Detector M-PAM.
        
        Returns
        -------
        x_hat : M-PAM detected symbols.
        
        '''
        y_k = y_k.reshape(-1, order='F')        
        
        XA = y_k.reshape(-1,1)
        XB = self.x.reshape(-1,1)
        
        # Computing pair-wise Euclidean distances
        d = cdist(XA, XB, metric='euclidean')
        # Indices corresponding minimum Euclidian distance
        ind = np.argmin(d,axis=1)
        
        x_hat =  np.array([self.x[ind[i]] for i in range(len(ind))])       
        
        
        # x_hat = np.zeros(len(y_k))
        # for k in range(len(y_k)):
        #     if y_k[k]>=0:
        #         for l in range(self.M//2):
        #             if (y_k[k] >= self.d*l) and (y_k[k] < self.d*(l+1)):
        #                 x_hat[k] = self.x[int(self.M/2 + l)]
        #             elif y_k[k]>self.d*(l+1): 
        #                 x_hat[k] = self.x[-1]
        #     else:    
        #         for l in range(self.M//2):
        #             if (y_k[k] < -self.d*l) and (y_k[k] >= -self.d*(l+1)):
        #                 x_hat[k] = self.x[int(self.M/2 - l - 1)]
        #             elif y_k[k] < -self.d*(l+1):
        #                 x_hat[k] = self.x[0]
        
        return x_hat

    def plot_const(self):
        '''
        # Plot the M-PAM constellation.

        Returns
        -------
        None.

        '''
        
        fig, ax = plt.subplots(figsize=(7,1), dpi = 300)
        ax.scatter(self.x, np.zeros(len(self.x)), color = '#5050b7f1', s = 75)
        ax.set_yticks([])
        ax.set_xlabel('Em fase')
        ax.set_title('Constelação {}-PAM'.format(self.M))
        ax.grid()
        plt.show()
        
        pass

# ----------------------------------------------------------------------------

class QAM():
    '''
        M-QAM
    '''
    def __init__(self, M, Ex):
        self.M = M          # Constellation order
        self.Ex = Ex        # Average energy of the constellation
        self.b = np.log2(M) # Number of bits per symbol
        # Minimum distance between two symbos with average energy Ex
        self.d = np.sqrt(6 * Ex / (M - 1))
        
        if (M==1) or (np.mod(np.log2(M),2)!=0): # M not a even power of 2
            raise ValueError('M must be even power of 2')
                
    def mod(self):
        '''
        Generate M-QAM symbols.
        
        Returns
        -------
        x : M-QAM symbols.

        '''
        
        # M-QAM constellation
        self.x_pam = np.arange(-(np.sqrt(self.M)-1), np.sqrt(self.M), 2) * self.d/2
        
        self.x = (np.ones((int(np.sqrt(self.M)), 1)) @ self.x_pam.reshape(1,-1) + 
                  1j*self.x_pam.reshape(-1,1)[::-1] @ np.ones((1, int(np.sqrt(self.M)))))
        
        self.x = self.x.reshape(-1)
        
        return self.x
    
    def demod(self, y_k):
        '''
        Detector M-QAM.
        
        Returns
        -------
        x_hat : M-QAM detected symbols.
        
        '''
        
        y_k = y_k.reshape(-1, order='F')     
         
        XA = np.column_stack((y_k.real, y_k.imag))
        XB = np.column_stack((self.x.real, self.x.imag))
        
        # Computing pair-wise Euclidean distances
        d = cdist(XA, XB, metric='euclidean')
        # Indices corresponding minimum Euclidian distance
        ind = np.argmin(d, axis=1)
        
        x_hat =  np.array([self.x[ind[i]] for i in range(len(ind))])       
        
        return x_hat

    def plot_const(self):
        '''
        # Plot the M-PAM constellation.

        Returns
        -------
        None.

        '''
        
        fig, ax = plt.subplots(figsize=(6,6), dpi = 300)
        ax.scatter(self.x.real, self.x.imag, color = '#5050b7f1', s = 75)
        ax.set_xlabel('Em fase')
        ax.set_xlabel('Quadratura')
        ax.set_title('Constelação {}-QAM'.format(self.M))
        ax.grid()
        plt.show()
        
        pass
    
# ----------------------------------------------------------------------------

class PSK():
    '''
        M-PSK
    '''
    def __init__(self, M, theta_0):
        self.M = M             # Constellation order
        self.b = np.log2(M)    # Number of bits per symbol
        self.theta_0 = theta_0 # Initial fase
           
    def mod(self):
        '''
        Generate M-PSK symbols.
        
        Returns
        -------
        x : M-PSK symbols.

        '''
        
        m = np.arange(1,self.M+1)        
        theta = self.theta_0 + 2*np.pi*(m-1)/self.M
        
        # M-PSK constellation
        self.x = np.cos(theta) + 1j*np.sin(theta)
        
        self.x = self.x.reshape(-1)
        
        return self.x
    
    def demod(self, y_k):
        '''
        Detector M-QAM.
        
        Returns
        -------
        x_hat : M-PSK detected symbols.
        
        '''
        
        y_k = y_k.reshape(-1, order='F')     
     
        XA = np.column_stack((y_k.real, y_k.imag))
        XB = np.column_stack((self.x.real, self.x.imag))
        
        # Computing pair-wise Euclidean distances
        d = cdist(XA, XB, metric='euclidean')
        # Indices corresponding minimum Euclidian distance
        ind = np.argmin(d,axis=1)
        
        x_hat =  np.array([self.x[ind[i]] for i in range(len(ind))])       
        
        return x_hat

    def plot_const(self):
        '''
        # Plot the M-PAM constellation.

        Returns
        -------
        None.

        '''
        
        fig, ax = plt.subplots(figsize=(6,6), dpi = 300)
        ax.scatter(self.x.real, self.x.imag, color = '#5050b7f1', s = 75)
        ax.set_xlabel('Em fase')
        ax.set_xlabel('Quadratura')
        ax.set_title('Constelação {}-PSK'.format(self.M))
        ax.grid()
        plt.show()
        
        pass
    
# ----------------------------------------------------------------------------   

class OFDM():

    def __init__(self, N, Ncp, mapping):
        self.map = mapping               # map = 1 -> DMT / map = 0 -> OFDM
        self.N = N + N*self.map          # Comprimento do bloco de símbolo
        self.Ncp = Ncp                   # Comprimento do prefíxo cíclico
        
        if (mapping != 1) and (mapping != 0): 
            raise ValueError('mapping deve ser igual a 1 ou 0')
            
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
    
    def demod(self, y_n, h):
        """
        Realiza a demodulação DMT e OFDM.
    
        Entrada
        ----------
        y_n : Vetor de bloco de símbolos DMT/OFDM concatenados.
        h : Resposta ao impulso
    
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
        H = np.fft.fft(h, self.N)
        
        # Equalização
        for i in range(Ns):
            Y_map[:,i] = Y_map[:,i]/H
        
        # Demapeamento
        if self.map ==1:  
             Y = np.vstack((Y_map[1:self.N//2], 
                           np.real(Y_map[0]) + 1j*np.real(Y_map[self.N//2])))
        else:
            Y = Y_map
    
        return Y
    
    def waterFilling_RA(gamma_k, gap, Pt):
        """
        Realiza a alocação de potência ótima para o esquema DMT/OFDM    

        Parameters
        ----------
        gamma_k : vetor coluna com a nSNR normalizada de cada subportadora
        gap : valor do gap (sem ser em dB). 8.8 dB -> 1e-6 SER
        Pt : potencia total disponivel a ser distribuido pelas subportadoras

        Returns
        -------
        Pk : vetor coluna de mesmo tamanho que gamma_k com a potencia destinada a cada subportado

        """

        Pk_sort = np.zeros(np.shape(gamma_k))
        Pk = np.zeros(np.shape(gamma_k))
        gamma_kS = np.flipud(np.sort(gamma_k))
        index = np.flipud(np.argsort(gamma_k))
        gapGammakS = gap/gamma_kS
        kk = len(gamma_kS)
        Delta = Pt + sum(gapGammakS)
        lmd = kk/Delta
        while ((1/lmd) - gapGammakS[-1]) < 0:
            kk = kk - 1
            gapGammakS = gapGammakS[0:-1]
            Delta = Pt + sum(gapGammakS)
            lmd = kk/Delta 
        Pk_sort[0:len(gapGammakS)] = (1/lmd) - gapGammakS
        Pk[index] = Pk_sort
        
        return Pk