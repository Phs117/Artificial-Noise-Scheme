# -*- coding: utf-8 -*-
"""

"""
# Bibliotecas

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from CommPyAC.modulacao import QAM, OFDM
#from CommPyAC.pls import Jamming

from tqdm import tqdm


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
                d = np.random.randn(self.Ncp)
            else:
                d = (np.random.randn(self.Ncp) + 1j*np.random.randn(self.Ncp))*1/2
            a_i = Vn@d # vetor AN 
            Eai = np.linalg.norm(a_i)**2/len(a_i)
            a_i = np.sqrt(Ea) * a_i / np.sqrt(Eai)                   
            a = np.concatenate((a, a_i))        
        return a 


Lh = 300
# Canal de Bob
HB = np.load('H_PLC.npy')[:,6014]
hB = np.fft.ifft(HB)[:Lh]

# Canal de Eve perto
HEs = np.load('H_SP.npy')[:,9368] 
hEs = np.fft.ifft(HEs)[:Lh]

# Canal de Eve longe
HEl = np.load('H_LP.npy')[:,3437] 
hEl = np.fft.ifft(HEl)[:Lh]



#%%
# Constantes
M = 4
N = 2048
Ncp = 512
Nsymb = 10**1
Ns = Nsymb//N

# Variaveis
alphas = np.array([0,0.3, 0.50, 0.90])
Pot_dBm = np.arange(0,55,5)
Pot_linear = 10**((Pot_dBm-30)/10)

# Ruido artificial
an = Jamming(N, Ncp, mapping = 1)

# Canal AWGN modulado por PV
Pv = 10**-8

# Array para calculo do erro por simbolo
ser_Bob = np.zeros(len(Pot_dBm))
ser_Eve_short = np.zeros(len(Pot_dBm))
ser_Eve_long = np.zeros(len(Pot_dBm))

# Array de linestyle
lines = ['-','--','-.',':']
# Cores Bob e Eve,respectivamente

#AZUL #2737D8
#ROSA #D82790

# Criando uma grade de subplots com 1 linha e 2 colunas e tamanho 10 por 6
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,6))

for i,alpha in enumerate(alphas):    
    
    for k,Pot in tqdm(enumerate(Pot_linear),total = len(Pot_linear),desc = f'Simulaçao para alpha = {alpha}'):
        
        # Potencias moduladas
        Px = Pot * (1 - alpha)
        Pa = Pot * alpha
        #print(Pa) # Potencia de PA
        
        # Ruido Artifical
        a = an.an_cp(hB, Ns, Pa) 
        #print(np.linalg.norm(a[:,1])**2/(2*N+Ncp)) # Potencia de a
        
        # Geração de simbolos QAM
        qam = QAM(M, Px)
        X = qam.mod()
        X = np.random.choice(X, (N, Ns))
        ofdm = OFDM(N, Ncp, mapping = 1)
        x = ofdm.mod(X)
        x = x + a
        
        # Simbolos a serem transmitidos
        r_bob = np.convolve(x, hB)[:len(x)]
        r_eve_short = np.convolve(x, hEs)[:len(x)]
        r_eve_long = np.convolve(x, hEl)[:len(x)]
        
        # Canal AWGN
        y_bob = r_bob + (np.sqrt(Pv) * np.random.randn(len(r_bob)))
        y_eve_short = r_eve_short + (np.sqrt(Pv) * np.random.randn(len(r_eve_short)))
        y_eve_long = r_eve_long + (np.sqrt(Pv) * np.random.randn(len(r_eve_long)))
        
        # Calculo do erro
        Y_bob = ofdm.demod(y_bob, hB)
        Y_eve_short = ofdm.demod(y_eve_short, hEs)
        Y_eve_long = ofdm.demod(y_eve_long, hEl)
        
        X_hat = qam.demod(Y_bob)
        Xe_hat_short = qam.demod(Y_eve_short)
        Xe_hat_long = qam.demod(Y_eve_long)
        
        erro = sum(X.reshape(-1, order='F') != X_hat)
        erroe_short = sum(X.reshape(-1, order='F') != Xe_hat_short)
        erroe_long = sum(X.reshape(-1, order='F') != Xe_hat_long)
        
        ser_Bob[k] = erro/Nsymb 
        ser_Eve_short[k] = erroe_short/Nsymb
        ser_Eve_long[k] = erroe_long/Nsymb
    
    # Para cada alpha, plotar a curva da BER em função da potencia em dBm    
           
    # Plotando a BER de Bob na figura 1 
    ax1.semilogy(Pot_dBm,ser_Bob/np.log2(M),linestyle = lines[i], marker='s',color = '#2737D8')
    # Plotando a BER de Eve short path na figura 1
    ax1.semilogy(Pot_dBm,ser_Eve_short/np.log2(M),linestyle = lines[i], marker='o',color = '#D82790')
      
    # Plotando a BER de Bob na figura 2
    ax2.semilogy(Pot_dBm,ser_Bob/np.log2(M),linestyle = lines[i], marker='s',color = '#2737D8')
    # Plotando a BER de Eve long path na figura 2
    ax2.semilogy(Pot_dBm,ser_Eve_long/np.log2(M),linestyle = lines[i], marker='o',color = '#D82790')
    
# Plotando a figura

# FIGURA 1
# ax1.set_title("Bob & Short Eve")
ax1.set_xlabel('Pt [dBm]')
ax1.set_ylabel('BER')
ax1.grid()
ax1.set_ylim(10**-5,1)


# FIGURA 2
# ax2.set_title("Bob & Long Eve")
ax2.set_xlabel('Pt [dBm]')
ax2.set_ylabel('BER')
ax2.grid()
ax2.set_ylim(10**-5,1)

# Legenda da figura
custom_legend = [
    mlines.Line2D([], [], color='#2737D8', marker = 's',linestyle='none'),
    mlines.Line2D([], [], color='#D82790', marker = 'o',linestyle='none'),
    mlines.Line2D([], [], color='grey', linestyle='-'),
    mlines.Line2D([], [], color='grey', linestyle='--'),
    mlines.Line2D([], [], color='grey', linestyle='-.'),
    mlines.Line2D([], [], color='grey', linestyle=':')
]

# Adicionando a legenda na figura
fig.legend(custom_legend, ['Bob','Eve', r'$\alpha$ = 0',r'$\alpha$ = 0.3',r'$\alpha$ = 0.5',r'$\alpha$ = 0.9'],  bbox_to_anchor=(0.5,1.00), loc='upper center',frameon =False, ncol =6 ,fontsize = 12)

# Gerando o gráfico e salvando
plt.savefig('C:/Users\phs20\OneDrive\Documentos\IC\Gráficos\BER_Bob&Eve2.eps', format='eps', bbox_inches='tight')
plt.show()
