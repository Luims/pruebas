import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

# Función para calcular el Sortino Ratio
def sortino(returns, risk_free_rate=0.02, target_return=0):
    excess_returns = returns - risk_free_rate / 252  # Asumiendo retornos diarios
    downside_returns = excess_returns[excess_returns < target_return]
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    return np.sqrt(252) * excess_returns.mean() / downside_deviation
  
#estadisticas
def estadisticas(emisora):
    # Cuerpo de la función
    resultado = np.array([np.mean(emisora), np.std(emisora),sharpe_ratio(emisora), sortino(emisora),skew(emisora), kurtosis(emisora) ])  # Una operación simple
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

#DRAWDOWN
def obtener_datos_acciones(simbolos, start_date, end_date):
    """Descarga datos históricos de precios"""
    data = yf.download(simbolos, start=start_date, end=end_date)['Close']
    return data.ffill().dropna()

def calcular_drawdown(precios):
    """
    Calcula el drawdown y high water mark
    Retorna los valores en decimales (no en porcentaje)
    """
    high_water_mark = precios.expanding().max()
    drawdown = (precios - high_water_mark) / high_water_mark
    return drawdown, high_water_mark

def graficar_drawdown_financiero(precios, titulo="Análisis de Drawdown"):
    """Crea gráfico de precios y drawdown en subplots"""
    drawdown, hwm = calcular_drawdown(precios)

    # Crear figura con subplots
    fig = make_subplots(rows=2, cols=1,
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       row_heights=[0.7, 0.3])

    # Subplot 1: Precios y HWM
    fig.add_trace(
        go.Scatter(
            x=precios.index,
            y=precios.values,
            name='Precio',
            line=dict(color='blue'),
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=hwm.index,
            y=hwm.values,
            name='High Water Mark',
            line=dict(color='green', dash='dash'),
        ),
        row=1, col=1
    )

    # Subplot 2: Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name='Drawdown',
            line=dict(color='red'),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)'
        ),
        row=2, col=1
    )

    # Línea horizontal en 0 para el drawdown
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        row=2
    )

    # Actualizar layout
    fig.update_layout(
        title=titulo,
        height=800,  # Altura total de la figura
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified'
    )

    # Actualizar ejes Y
    fig.update_yaxes(title="Precio", row=1, col=1)
    fig.update_yaxes(
        title="Drawdown %",
        tickformat=".1%",
        range=[-1, 0.1],  # Límites del drawdown entre -100% y 10%
        row=2,
        col=1
    )

    # Actualizar eje X
    fig.update_xaxes(title="Fecha", row=2, col=1)

    return fig

def obtener_max_drawdown_info(precios):
    """Obtiene información detallada del máximo drawdown"""
    drawdown, _ = calcular_drawdown(precios)

    max_drawdown = drawdown.min()
    fecha_max_drawdown = drawdown.idxmin()
    pico_anterior = precios[:fecha_max_drawdown].idxmax()

    datos_posteriores = drawdown[fecha_max_drawdown:]
    fechas_recuperacion = datos_posteriores[datos_posteriores >= 0]
    fecha_recuperacion = fechas_recuperacion.index[0] if len(fechas_recuperacion) > 0 else None

    duracion_caida = (fecha_max_drawdown - pico_anterior).days
    duracion_recuperacion = (fecha_recuperacion - fecha_max_drawdown).days if fecha_recuperacion else None
    duracion_total = (fecha_recuperacion - pico_anterior).days if fecha_recuperacion else None

    return {
        'max_drawdown': max_drawdown * 100,
        'fecha_pico': pico_anterior,
        'fecha_valle': fecha_max_drawdown,
        'fecha_recuperacion': fecha_recuperacion,
        'duracion_caida': duracion_caida,
        'duracion_recuperacion': duracion_recuperacion,
        'duracion_total': duracion_total
    }
#Funcion a usar 
def drawdown(simbolo, start_date,end_date):
# Obtener datos
    datos = obtener_datos_acciones(simbolo, start_date, end_date)

# Si los datos son para múltiples símbolos, seleccionar uno
    if isinstance(datos, pd.DataFrame):
        precios = datos[simbolo]
    else:
        precios = datos

# Graficar
    fig = graficar_drawdown_financiero(precios, f'Análisis de Drawdown - {simbolo}')
    st.plotly_chart(fig)

# Obtener información detallada
    info_dd = obtener_max_drawdown_info(precios)

# Imprimir resultados
    st.text(f"\nAnálisis de Drawdown para {simbolo}:")
    st.text(f"Máximo Drawdown: {info_dd['max_drawdown']:.2f}%")
    st.text(f"Fecha del pico: {info_dd['fecha_pico'].strftime('%Y-%m-%d')}")
    st.text(f"Fecha del valle: {info_dd['fecha_valle'].strftime('%Y-%m-%d')}")
    st.text(f"Duración de la caída: {info_dd['duracion_caida']} días")

    if info_dd['fecha_recuperacion'] is not None:
        st.text(f"Fecha de recuperación: {info_dd['fecha_recuperacion'].strftime('%Y-%m-%d')}")
        st.text(f"Duración de la recuperación: {info_dd['duracion_recuperacion']} días")
        st.text(f"Duración total: {info_dd['duracion_total']} días")
    else:
        st.text("El activo aún no se ha recuperado del máximo drawdown")

#Graficar rendimientos
def grafica_ren(df,emisora):
    fig = px.line(
        df,
        x='Date',
        y=emisora,
        title="Rendimientos de la Acción",
        labels={'fecha': "Fecha", 'rendimientos': "Rendimiento"},
        template="plotly_white"
    )

# Personalizar el gráfico
    fig.update_layout(
        title_font=dict(size=20, family='Arial', color='darkblue'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        hovermode="x unified"
    )

# Mostrar el gráfico
    fig.show()
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
    col1, col2 = st.columns([2,1])
    if activo_seleccionado == "Activo 1":
      with col1:
        st.write("### Gráfica de Métricas")
        fig = px.line(
        df,
        x='Date',
        y='AGUA.MX_rend',
        title="Rendimientos de la Acción",
        labels={'fecha': "Fecha", 'rendimientos': "Rendimiento"},
        template="plotly_white")
# Personalizar el gráfico
        fig.update_layout(
        title_font=dict(size=20, family='Arial', color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        hovermode="x unified")

        st.plotly_chart(fig)

        
    # Columna derecha: Estadísticas en tabla
      with col2:
        st.write("### Datos del Activo Seleccionado")
        st.text(estadisticas(df['AGUA.MX_rend']))

      simbolo = 'AGUA.MX'
      start_date = '2010-01-01'
      end_date = datetime.now()
      drawdown(simbolo, start_date,end_date)

  if activo_seleccionado == "Activo 2":
      with col1:
        st.write("### Gráfica de Métricas")
        fig = px.line(
        df,
        x='Date',
        y='AMZN.MX_rend',
        title="Rendimientos de la Acción",
        labels={'fecha': "Fecha", 'rendimientos': "Rendimiento"},
        template="plotly_white")
# Personalizar el gráfico
        fig.update_layout(
        title_font=dict(size=20, family='Arial', color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        hovermode="x unified")
        st.plotly_chart(fig)
    # Columna derecha: Estadísticas en tabla
      with col2:
        st.write("### Datos del Activo Seleccionado")
        st.text(estadisticas(df['AMZN.MX_rend']))

      simbolo = 'AMZN.MX'
      start_date = '2010-01-01'
      end_date = datetime.now()
      drawdown(simbolo, start_date,end_date)
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
