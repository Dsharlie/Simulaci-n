# -*- coding: utf-8 -*-
"""
**Librerías**
"""

import math
import numpy as np
import statistics as st
from scipy.stats import expon, norm, lognorm, gamma, weibull_min, beta, uniform, chi2, triang
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

"""**Gráficas PP plot y QQ plot para una distribución normal**"""
def PP_QQ_plot_normal(data,media="estimado",desvesta="estimado"):
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if media=="estimado":
        mean = np.mean(data)
        print("Parámetro estimado: Media = "+str(mean))
    else:
        mean = media
    if desvesta=="estimado":
        std_dev = np.std(data)
        print("Parámetro estimado: Desviación Estándar = "+str(std_dev))
    else:
        std_dev = desvesta
    
    n = len(data)
    # Se crea una Q-Q plot con el paquete statsmodels.api para la distribución normal
    sm.qqplot(data, norm, loc=mean, scale=std_dev, line='45')
    # Se agrega un título a la gráfica
    plt.title("Q-Q Plot")
    # Se muestra la gráfica
    plt.show()
    
    fig, ax = plt.subplots()
    # Se calculan las probabilidades empíricas
    p = np.arange(1, n + 1) / n - 0.5 / n
    # Se calculan las probabilidades teóricas
    pp = np.sort(norm.cdf(data,loc=mean,scale=std_dev))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=ax)
    ax.set_title('P-P plot')
    ax.set_xlabel('Theoretical Probabilities')
    ax.set_ylabel('Sample Probabilities')
    ax.margins(x=0, y=0)
    # Se dibuja la línea roja de 45°
    plt.plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    # Se muestra la gráfica
    plt.show()
    
"""**Gráficas PP plot y QQ plot para una distribución lognormal**"""
def PP_QQ_plot_lognormal(data,media="estimado",desvesta="estimado"):
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if media=="estimado":
        mean = np.mean(np.log(data))
        print("Parámetro estimado (normal asociada): Media = "+str(mean))
    else:
        mean = media
    
    if desvesta=="estimado":
        std_dev = np.std(np.log(data))
        print("Parámetro estimado (normal asociada): Desviación Estándar = "+str(std_dev))
    else:
        std_dev = desvesta  
    
    n = len(data)
    sm.qqplot(data, lognorm, distargs=(std_dev,), scale=np.exp(mean), line='45')
    # Se agrega un título a la gráfica
    plt.title("Q-Q Plot")
    # Se muestra la gráfica
    plt.show()
    
    fig, ax = plt.subplots()
    # Se calculan las probabilidades empíricas
    p = np.arange(1, n + 1) / n - 0.5 / n
    # Se calculan las probabilidades teóricas
    pp = np.sort(lognorm.cdf(data,s=std_dev,scale=np.exp(mean)))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=ax)
    ax.set_title('P-P plot')
    ax.set_xlabel('Theoretical Probabilities')
    ax.set_ylabel('Sample Probabilities')
    ax.margins(x=0, y=0)
    # Se dibuja la línea roja de 45°
    plt.plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    # Se muestra la gráfica
    plt.show()

"""**Gráficas PP plot y QQ plot para una distribución exponencial**"""
def PP_QQ_plot_exponential(data,tasa="estimado"):
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if tasa=="estimado":
        mean = np.mean(data)
        print("Parámetro estimado: Tasa = "+str(1/mean))
    else:
        mean = 1/tasa
    
    n = len(data)
    sm.qqplot(data, expon, scale=mean, line='45')
    # Se agrega un título a la gráfica
    plt.title("Q-Q Plot")
    # Se muestra la gráfica
    plt.show()
    
    fig, ax = plt.subplots()
    # Se calculan las probabilidades empíricas
    p = np.arange(1, n + 1) / n - 0.5 / n
    # Se calculan las probabilidades teóricas
    pp = np.sort(expon.cdf(data,scale=mean))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=ax)
    ax.set_title('P-P plot')
    ax.set_xlabel('Theoretical Probabilities')
    ax.set_ylabel('Sample Probabilities')
    ax.margins(x=0, y=0)
    # Se dibuja la línea roja de 45°
    plt.plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    # Se muestra la gráfica
    plt.show()

"""**Gráficas PP plot y QQ plot para una distribución uniforme**"""
def PP_QQ_plot_uniform(data,minimo="estimado",maximo="estimado"):
    n = len(data)
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if minimo=="estimado":
        a = np.min(data)
        print("Parámetro estimado: a = "+str(a))
    else:
        a = minimo
    
    if maximo=="estimado":
        b = np.max(data)
        print("Parámetro estimado: b = "+str(b))
    else:
        b = maximo

    sm.qqplot(data,uniform,loc=a,scale=b-a,line='45')
    # Se agrega un título a la gráfica
    plt.title("Q-Q Plot")
    # Se muestra la gráfica
    plt.show()
    
    fig, ax = plt.subplots()
    # Se calculan las probabilidades empíricas
    p = np.arange(1, n + 1) / n - 0.5 / n
    # Se calculan las probabilidades teóricas
    pp = np.sort(uniform.cdf(data,loc=a,scale=b-a))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=ax)
    ax.set_title('P-P plot')
    ax.set_xlabel('Theoretical Probabilities')
    ax.set_ylabel('Sample Probabilities')
    ax.margins(x=0, y=0)
    # Se dibuja la línea roja de 45°
    plt.plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    # Se muestra la gráfica
    plt.show()

"""**Gráficas PP plot y QQ plot para una distribución triangular**"""
def PP_QQ_plot_triangular(data,minimo="estimado",maximo="estimado",moda="estimado"):
    n = len(data)
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if minimo=="estimado":
        a = np.min(data)
        print("Parámetro estimado: a = "+str(a))
    else:
        a = minimo

    if maximo=="estimado":
        b = np.max(data)
        print("Parámetro estimado: b = "+str(b))
    else:
        b = maximo
        
    if moda=="estimado":
        c = st.mode(data)
        print("Parámetro estimado: c = "+str(c))
    else:
        c = moda
    
    sm.qqplot(data,triang,distargs=((c - a)/(b - a),),loc=a,scale=b-a,line='45')
    # Se agrega un título a la gráfica
    plt.title("Q-Q Plot")
    # Se muestra la gráfica
    plt.show()
    
    fig, ax = plt.subplots()
    # Se calculan las probabilidades empíricas
    p = np.arange(1, n + 1) / n - 0.5 / n
    # Se calculan las probabilidades teóricas
    pp = np.sort(triang.cdf(data,c=(c - a)/(b - a),loc=a,scale=b-a))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=ax)
    ax.set_title('P-P plot')
    ax.set_xlabel('Theoretical Probabilities')
    ax.set_ylabel('Sample Probabilities')
    ax.margins(x=0, y=0)
    # Se dibuja la línea roja de 45°
    plt.plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    # Se muestra la gráfica
    plt.show()

"""**Gráficas PP plot y QQ plot para una distribución gamma**"""
def PP_QQ_plot_gamma(data,media="estimado",varianza="estimado"):
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if media=="estimado":
        mean = np.mean(data)
        print("Parámetro estimado: Media = "+str(mean))
    else:
        mean = media
    
    if varianza=="estimado":
        var = np.var(data)
        print("Parámetro estimado: Varianza = "+str(var))
    else:
        var = varianza
    
    n = len(data)
    sm.qqplot(data,gamma,distargs=(var/mean,),scale=mean,line='45')
    # Se agrega un título a la gráfica
    plt.title("Q-Q Plot")
    # Se muestra la gráfica
    plt.show()
    
    fig, ax = plt.subplots()
    # Se calculan las probabilidades empíricas
    p = np.arange(1, n + 1) / n - 0.5 / n
    # Se calculan las probabilidades teóricas
    pp = np.sort(gamma.cdf(data,a=var/mean, scale=mean))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=ax)
    ax.set_title('P-P plot')
    ax.set_xlabel('Theoretical Probabilities')
    ax.set_ylabel('Sample Probabilities')
    ax.margins(x=0, y=0)
    # Se dibuja la línea roja de 45°
    plt.plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    # Se muestra la gráfica
    plt.show()

"""**Gráficas PP plot y QQ plot para una distribución weibull**"""
def PP_QQ_plot_weibull(data,forma="estimado",escala="estimado"):
    shape, loc, scale = weibull_min.fit(data, floc=0)
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if escala!="estimado":
        scale = escala
    else:
        print("Parámetro estimado: Escala = "+str(scale))
    
    if forma!="estimado":
        shape = forma
    else:
        print("Parámetro estimado: Forma = "+str(shape))
    
    n = len(data)
    sm.qqplot(data,weibull_min,distargs=(shape,),loc=loc,scale=scale,line='45')
    # Se agrega un título a la gráfica
    plt.title("Q-Q Plot")
    # Se muestra la gráfica
    plt.show()
    
    fig, ax = plt.subplots()
    # Se calculan las probabilidades empíricas
    p = np.arange(1, n + 1) / n - 0.5 / n
    # Se calculan las probabilidades teóricas
    pp = np.sort(weibull_min.cdf(data,shape,loc=loc,scale=scale))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=ax)
    ax.set_title('P-P plot')
    ax.set_xlabel('Theoretical Probabilities')
    ax.set_ylabel('Sample Probabilities')
    ax.margins(x=0, y=0)
    # Se dibuja la línea roja de 45°
    plt.plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    # Se muestra la gráfica
    plt.show()