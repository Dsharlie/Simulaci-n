# -*- coding: utf-8 -*-

import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import expon, norm, lognorm, gamma, weibull_min, beta, uniform, triang, kstest

"""**Prueba KS para una distribución normal**"""
def KS_test_normal(data,media=0,desvesta=1):
    """Performs a KS test for a normal probability distribution.

    Arguments:
    data -- a list of data values
    media -- Mean
    desvesta -- Standard Deviation

    Returns:
    A tuple containing the test statistic and p-value.
    """
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    mean = media
    std_dev = desvesta
    
    ks_statistic, p_value = kstest(data,norm.cdf,args=(mean,std_dev))

    respuesta = "Kolmogorov Smirnov statistic: "+ str(ks_statistic) + "\np-value: " + str(p_value)

    return print(respuesta)

"""**Prueba KS para una distribución lognormal**"""
def KS_test_lognormal(data,media=0,desvesta=1):
    """Performs a KS test for a lognormal probability distribution.

    Arguments:
    data -- a list of data values
    media -- Mean (normal asociada)
    desvesta -- Standard Deviation (normal asociada)

    Returns:
    A tuple containing the test statistic and p-value.
    """
    mean = media
    std_dev = desvesta
    
    ks_statistic, p_value = kstest(data,lognorm.cdf,args=(std_dev,np.exp(mean)))

    respuesta = "Kolmogorov Smirnov statistic: "+ str(ks_statistic) + "\np-value: " + str(p_value)
    
    return print(respuesta)

"""**Prueba KS para una distribución exponencial**"""
def KS_test_exponential(data,tasa):
  
    """Performs a KS test for an exponential probability distribution.

    Arguments:
    data -- a list of data values
    tasa -- rate

    Returns:
    A tuple containing the test statistic and p-value.
    """
    mean = 1/tasa
    
    ks_statistic, p_value = kstest(data, 'expon', args=(0, mean))

    respuesta = "Kolmogorov Smirnov statistic: "+ str(ks_statistic) + "\np-value: " + str(p_value)
    
    return print(respuesta)

"""**Prueba KS para una distribución uniforme**"""
def KS_test_uniform(data,minimo=0,maximo=1):
    """Performs a KS test for a uniform probability distribution.

    Arguments:
    data -- a list of data values
    a -- the lower bound of the uniform distribution
    b -- the upper bound of the uniform distribution

    Returns:
    A tuple containing the test statistic and p-value.
    """
    a = minimo
    b = maximo

    ks_statistic, p_value = kstest(data,uniform.cdf,args=(a,b-a))

    respuesta = "Kolmogorov Smirnov statistic: "+ str(ks_statistic) + "\np-value: " + str(p_value)
    
    return print(respuesta)

"""**Prueba KS para una distribución triangular**"""
def KS_test_triangular(data,minimo=0,maximo=1,moda=0.5):
    """Performs a KS test for a triangular probability distribution.

    Arguments:
    data -- a list of data values
    a -- the lower bound of the triangular distribution
    b -- the upper bound of the triangular distribution
    c -- the mode of the triangular distribution

    Returns:
    A tuple containing the test statistic, p-value
    """
    a = minimo
    b = maximo
    c = moda

    ks_statistic, p_value = kstest(data,triang.cdf,args=((c-a)/(b-a),a,b-a))

    respuesta = "Kolmogorov Smirnov statistic: "+ str(ks_statistic) + "\np-value: " + str(p_value)
    
    return print(respuesta)

"""**Prueba KS para una distribución gamma**"""
def KS_test_gamma(data,media=0,varianza=1):
    """Performs a KS test for a gamma probability distribution.

    Arguments:
    data -- a list of data values
    media -- Mean
    varianza -- Variance

    Returns:
    A tuple containing the test statistic and p-value.
    """
    mean = media
    var = varianza
    
    ks_statistic, p_value = kstest(data,gamma.cdf,args=(var/mean,0,mean))

    respuesta = "Kolmogorov Smirnov statistic: "+ str(ks_statistic) + "\np-value: " + str(p_value)
    
    return print(respuesta)

"""**Prueba KS para una distribución weibull**"""
def KS_test_weibull(data,forma=1,escala=1):
    """Performs a KS test for a Weibull probability distribution.

    Arguments:
    data -- a list of data values
    forma -- Data shape factor
    escala -- Data scale factor

    Returns:
    A tuple containing the test statistic and p-value.
    """
    scale = escala
    shape = forma
    
    ks_statistic, p_value = kstest(data,weibull_min.cdf,args=(shape,0,scale))

    respuesta = "Kolmogorov Smirnov statistic: "+ str(ks_statistic) + "\np-value: " + str(p_value)
    
    return print(respuesta)

# Se compara con una función definida por el usuario
def PP_plot_custom(data,function):
  fig, ax = plt.subplots()
  n = len(data)
  # Se calculan las probabilidades empíricas
  p = np.arange(1, n + 1) / n - 0.5 / n
  # Se calculan las probabilidades teóricas
  pp = np.sort(function)
  sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=ax)
  ax.set_title('P-P plot')
  ax.set_xlabel('Theoretical Probabilities')
  ax.set_ylabel('Sample Probabilities')
  ax.margins(x=0, y=0)
  # Se dibuja la línea roja de 45°
  plt.plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
  # Se muestra la gráfica
  plt.show()

def KS_custom(data,function):
  ks_statistic, p_value = kstest(data,lambda x:function)
  respuesta = "Kolmogorov Smirnov statistic: "+ str(ks_statistic) + "\np-value: " + str(p_value)
  return print(respuesta)