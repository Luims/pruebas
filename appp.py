import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import math as mt
from scipy.stats import norm
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import chi2
#Emisoras y fechas
emisoras = ['AGUA.MX','AMZN.MX', 'CHDRAUIB.MX', 'HD.MX','MELIN.MX']
fi = '2010-01-01'
ff = '2023-12-31'
df = yf.download(emisoras, start = fi, end = ff)['Close'].reset_index()
df2=df.copy()
# Configuración inicial de la app
st.set_page_config(page_title="Aplicación de Portafolios", layout="wide")

#Rendimientos
for emisora in emisoras:
  df[emisora + '_rend'] = np.log(df[emisora]/df[emisora].shift(252))
df.dropna(inplace =True)
df.reset_index(inplace = True, drop = True)

for i in df.columns[1:6]:
    df[i+'_Anual_rend']=np.log(df[i]/df[i].shift(1))*252
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

# Convertir la columna 'fecha' al tipo datetime si no lo está
df['Date'] = pd.to_datetime(df['Date'])

# Dividir el DataFrame
df_hasta_2020 = df.loc[:'2020-12-31']  # Hasta el final de 2020
df_desde_2020 = df.loc['2020-01-01':]  # Desde el inicio de 2020

rf= 0.1
mu = np.array([np.mean(df['AGUA.MX_rend']),np.mean(df['AMZN.MX_rend']), np.mean(df['CHDRAUIB.MX_rend']), np.mean(df['HD.MX_rend']),np.mean(df['MELIN.MX_rend'])])
sigma = np.array([np.std(df['AGUA.MX_rend']),np.std(df['AMZN.MX_rend']), np.std(df['CHDRAUIB.MX_rend']), np.std(df['HD.MX_rend']),np.std(df['MELIN.MX_rend'])])
sesgo  = np.array([skew(df['AGUA.MX_rend']),skew(df['AMZN.MX_rend']), skew(df['CHDRAUIB.MX_rend']), skew(df['HD.MX_rend']),skew(df['MELIN.MX_rend'])])
curtosis = np.array([kurtosis(df['AGUA.MX_rend']),kurtosis(df['AMZN.MX_rend']), kurtosis(df['CHDRAUIB.MX_rend']), kurtosis(df['HD.MX_rend']),kurtosis(df['MELIN.MX_rend'])])
matriz_Cov = df[['AGUA.MX_rend','AMZN.MX_rend','CHDRAUIB.MX_rend','HD.MX_rend','MELIN.MX_rend']].cov().values
matriz_Corr = df[['AGUA.MX_rend','AMZN.MX_rend','CHDRAUIB.MX_rend','HD.MX_rend','MELIN.MX_rend']].corr().values
#mu,sigma,sesgo,curtosis, matriz_Cov,matriz_Corr

# Función para calcular el Sharpe Ratio
def sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252  # Asumiendo retornos diarios
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

#estadisticas
def estadisticas(emsiora):
    # Cuerpo de la función
    resultado = np.array([np.mean(emisora), np.std(emisora),sharpe_ratio(emisora), sortino(emisora) ])  # Una operación simple
    return resultado

#black cock
def bl(prior, cov,t,view,q):
    vp = np.sqrt(np.transpose(prior) @ cov @ prior)
    print('-------------------------vp---------------------------------')
    print(vp)
    abersion = 0.5/vp
    print('------------------------abersion----------------------------------')
    print(abersion)
    vector_exceso_de_retorno = abersion*(cov @ prior)
    print('-------------------vector_exceso_de_retorno--------------------------------------')
    print(vector_exceso_de_retorno)
    sigmapi= (1/t)*cov
    print('-----------------------sigmapi-----------------------------------')
    print(sigmapi)
    omega=np.diag(np.diag(view @ (sigmapi) @ np.transpose(view)))
    print('---------------------------omega-------------------------------')
    print(omega)
    omega1 = np.linalg.inv(omega)
    sigamapi1 = np.linalg.inv(sigmapi)
    r = (np.linalg.inv(sigamapi1+(np.transpose(view) @ omega1 @ view))) @ ((sigamapi1 @ vector_exceso_de_retorno)+(np.transpose(view) @ omega1 @ q))
    print('---------------------------r-------------------------------')
    print(r)
    w = (1/2.24) * np.linalg.inv(cov) @ r # Una operación simple
    print('-----------peso--------------')
    return w  # Se retorna el resultado

def lagrange(ren, cov, ex):
    incov = np.linalg.inv(cov)
    uno=[]
    for i in range(len(ren)):
        uno.append(1)
    unot = np.transpose(np.array(uno))
    print('---------------1------------------')
    print(uno)
    A=  uno @ incov @ unot
    print('---------------a------------------')
    print(A)
    B = np.transpose(ren) @ incov @ unot
    print('---------------b------------------')
    print(B)
    C = np.transpose(ren) @ incov @ ren
    print('---------------c------------------')
    print(C)
    w =(1 / (A * C - B**2)) * incov @ ((A * ren - B * unot) * ex + (C * unot - B * ren))
    return w

def minima_varianza(cov):
    incov = np.linalg.inv(cov)
    uno=[]
    for i in range(len(cov)):
        uno.append(1)
    unot = np.transpose(np.array(uno))
    print('---------------1------------------')
    print(uno)
    A=  uno @ incov @ unot
    w = ((incov @ unot)/A)
    return w
#------------------------------------------------------------------------------------
# Barra de navegación
st.sidebar.title("Navegación")
pages = ["Portada", "Selección de Activos", "Estadística de Activos", "Portafolio 1", "Portafolio 2", "Black-Litterman"]
selection = st.sidebar.radio("Selecciona una página:", pages)

# Portada
if selection == "Portada":
    st.title("Bienvenido a la Aplicación de Portafolios")
    st.write("Esta aplicación te ayudará a gestionar y analizar activos financieros.")
    st.image("https://via.placeholder.com/800x400", caption="Imagen representativa")

# Selección de Activos
elif selection == "Selección de Activos":
    st.title("Selección de Activos")
    st.write("Ingresa información sobre los activos:")
    st.text_input("Activo 1:")
    st.text_input("Activo 2:")
    st.text_input("Activo 3:")
    st.text_input("Activo 4:")
    st.text_input("Activo 5:")

# Estadística de Activos
elif selection == "Estadística de Activos":
    st.title("Estadística de Activos")
    activos = ["Activo 1", "Activo 2", "Activo 3", "Activo 4", "Activo 5"]
    activo_seleccionado = st.selectbox("Selecciona un activo:", activos)
    if activo_seleccionado == "Activo 1":
      st.write(f"Mostrando estadísticas para: {activo_seleccionado}")
      st.write("Aquí se mostrarán las estadísticas relevantes del activo seleccionado.")

# Portafolio 1
elif selection == "Portafolio 1":
    st.title("Portafolio 1")
    portafolios = ["Portafolio A", "Portafolio B", "Portafolio C"]
    print(lagrange(mu, matriz_Cov,0.10))
    portafolio_seleccionado = st.selectbox("Selecciona un portafolio:", portafolios)
    st.write(f"Mostrando información para: {portafolio_seleccionado}")
    st.write("Aquí se mostrará información detallada del portafolio seleccionado.")

# Portafolio 2
elif selection == "Portafolio 2":
    st.title("Portafolio 2")
    portafolios = ["Portafolio X", "Portafolio Y", "Portafolio Z"]
    portafolio_seleccionado = st.selectbox("Selecciona un portafolio:", portafolios)
    st.write(f"Mostrando información para: {portafolio_seleccionado}")
    st.write("Aquí se mostrará información detallada del portafolio seleccionado.")

# Black-Litterman
elif selection == "Black-Litterman":
    st.title("Modelo Black-Litterman")
    
    st.write("Información sobre el modelo Black-Litterman será presentada aquí.")
