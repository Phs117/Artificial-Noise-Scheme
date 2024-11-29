# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 10:33:34 2021

@author: Ândrei Camponogara
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def rc(alpha, span, L):
    """
    Projeta o puso cosseno levantado.
    
    Entradas:
        alpha :  Fator de roll-off
        span : Número de símbolos que o pulso se espalha
        L : Fator de sobreamostragem (ou seja, cada símbolo contém L amostras)
    Saída:
        (t, p) : Base temporal t e o sinal p(t) como tupla 
        
    """
    
    t = np.arange(-span/2, span/2 + 1/L, 1/L) # +/- discrete-time base
    
    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.divide(np.sin(np.pi*t),(np.pi*t)) #assume Tsym=1
        B = np.divide(np.cos(np.pi*alpha*t),1-(2*alpha*t)**2)
        p = A*B
     
    # -----------------------------------    
    # Lidando com as singularidades
    
    # singularidade em p(t=0)
    p[np.argwhere(np.isnan(p))] = 1 
    # Singularidade em t = +/- T/(2alpha)
    p[np.argwhere(np.isinf(p))] = (alpha/2)*np.sin(np.divide(np.pi,(2*alpha)))
    
    
    # # ------------------------------------
    # # Atraso do filtro (ordem do filtro(N-1)/2)
    # atraso = int((len(p)-1)/2)
    
    return (t, p)


def srrc(beta, Tb, L, Nb, plot):
    '''
    Gera o pulso raiz quadrada do cosseno levantado.

    Entradas
    ----------
    beta : Fator de roll-off
    Tb : Período de símbolo
    L : Fator de sobreamostragem
    Nb : Número de símbolos que o pulso se espalha
    plot : Se igual a '1', o pulso gerado é plotado

    Saídas
    -------
    (t, p) : Base temporal t e o sinal p(t) como tupla    
    
    '''
    
    Fs = L/Tb
    Ts = 1/Fs
    
    t = np.arange(-Nb*Tb/2, Nb*Tb/2 + Ts, Ts)
    
    p = (1/Tb)*((np.sin(np.pi*t*(1-beta)/Tb) + (4*beta*t/Tb) * np.cos(np.pi*t*(1+beta)/Tb))/
                ((np.pi*t/Tb)*(1 - (4*beta*t/Tb)**2)))    
    
    p[t == 0] = (1/Tb)*(1 - beta + 4*beta/np.pi)
    
    p[t == Tb/(4*beta)] = (beta/(np.sqrt(2)*Tb))*((1+(2/np.pi))*np.sin(np.pi/(4*beta)) + (1-(2/np.pi))*np.cos(np.pi/(4*beta))) 
    
    p[t == -Tb/(4*beta)] = (beta/(np.sqrt(2)*Tb))*((1+(2/np.pi))*np.sin(np.pi/(4*beta)) + (1-(2/np.pi))*np.cos(np.pi/(4*beta))) 
    
    # # Atraso do filtro (ordem do filtro(N-1)/2)
    # atraso = int((len(p)-1)/2)
    
    if plot==1:
        # FFT pulso de transmissão
        P_f = fft(p)
        f = fftfreq(len(P_f), Tb/L)

        # Plote do pulso de transmissão p(t)
        fig, ax = plt.subplots(1,2,figsize=(12,4), dpi = 300)
        ax[0].plot(t*10**3,p, '#5050b7f1')
        ax[0].set_xlabel(r'$t$ [ms]')
        ax[0].set_ylabel(r'$x(t)$')
        ax[0].set_title('Filtro SRRC espalhado em {} símbolos'.format(Nb))
        ax[0].grid()
        
        ax[1].plot(f*10**-3,np.abs(P_f), '#5050b7f1')
        ax[1].set_xlabel(r'$f$ [kHz]')
        ax[1].set_ylabel(r'$|P(f)|$')
        ax[1].set_title('Resposta em frequência do filtro SRRC')
        ax[1].grid()
        plt.show()
    
    return (t, p)

def rect(A, fs, T, plot):
    """
    Gera o pulso retangular.

    Entradas:
        A : Amplitude do pulso retangular
        fs : Frequência de amostragem em Hz
        T : Duração do pulso em segundos (T < 1 segundo)

    Saídas:
        (t, p) : Base temporal t e o sinal p(t) como tupla 

    """
    
    t = np.arange(-0.5, 0.5, 1/fs)
    rect = (t > -T/2) * (t < T/2) + 0.5*(t == -T/2) + 0.5*(t == T/2) 
    
    p = A*rect
    
    if plot==1:
        # Plote do pulso de transmissão p(t)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(t*10**3, p, '#5050b7f1')
        ax.set_xlabel(r'$t$ [ms]')
        ax.set_ylabel(r'$p(t)$')
        ax.set_title('Pulso retangular')
        ax.grid()
        plt.show()
    
    return (t, p)

def diagramaOlho(xt,T,L):
    """
    Gera o diagrama do olho usando todas as partes de um dado sinal xt.
    
    Entradas:
    
        xt : float
             Sinal de informação recebido.
        T : float
            Intervalo de pulso.
        L : float
            Fator de sobreamostragem.
        
    Saídas:
    
        t_part: Array of float64
                Vetor temporal do olho.
        parts: Array of float64
               Matriz com os segmentos do sinal xt.       
        
    """
    Fs = L/T
    samples_perT = Fs*T
    samples_perWindow = 2*(Fs*T)
    parts = []
    startInd = 2*samples_perT   # Ignora alguns efeitos transientes no inicio
                                # do sinal
    
    for k in range(int(len(xt)/samples_perT) - 6):
        parts.append(xt[(startInd + k*samples_perT + 
                         np.arange(samples_perWindow)).astype(int)])
    parts = np.array(parts).T
    
    t_part = np.arange(-T, T, 1/Fs)
    
    # Figura
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(t_part, parts, '#5050b7f1', alpha=.95)
    ax.set_title(r'Diagrama do Olho')
    ax.grid()
    plt.show()
    
    return t_part, parts

