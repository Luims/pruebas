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
from numpy import *
from numpy.linalg import multi_dot
import scipy.optimize as sco
#Emisoras y fechas
emisoras = ['IEF','CETETRC.MX', 'SPY', 'EZA','IAU','^GSPC']
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
#df.reset_index(inplace = True, drop = True)
#df.set_index("Date", inplace=True)
for i in df.columns[1:7]:
    df[i+'_Anual_rend']=np.log(df[i]/df[i].shift(1))*252
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

# Convertir la columna 'fecha' al tipo datetime si no lo está
df['Date'] = pd.to_datetime(df['Date'])
df.set_index(df['Date'], inplace = True) 

# Dividir el DataFrame
df_hasta_2020 = df[df["Date"] < "2020-01-01"]  # Hasta el final de 2020
df_desde_2020 = df[df["Date"] >= "2020-01-01"]  # Desde el inicio de 2020

rf= 0.1
mu = np.array([np.mean(df['IEF_rend']),np.mean(df['CETETRC.MX_rend']), np.mean(df['SPY_rend']), np.mean(df['EZA_rend']),np.mean(df['IAU_rend'])])
sigma = np.array([np.std(df['IEF_rend']),np.std(df['CETETRC.MX_rend']), np.std(df['SPY_rend']), np.std(df['EZA_rend']),np.std(df['IAU_rend'])])
sesgo  = np.array([skew(df['IEF_rend']),skew(df['CETETRC.MX_rend']), skew(df['SPY_rend']), skew(df['EZA_rend']),skew(df['IAU_rend'])])
curtosis = np.array([kurtosis(df['IEF_rend']),kurtosis(df['CETETRC.MX_rend']), kurtosis(df['SPY_rend']), kurtosis(df['EZA_rend']),kurtosis(df['IAU_rend'])])
matriz_Cov = df[['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend']].cov().values
matriz_Corr = df[['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend']].corr().values
#mu,sigma,sesgo,curtosis, matriz_Cov,matriz_Corr
mu1 = np.array([np.mean(df_hasta_2020['IEF_rend']),np.mean(df_hasta_2020['CETETRC.MX_rend']), np.mean(df_hasta_2020['SPY_rend']), np.mean(df_hasta_2020['EZA_rend']),np.mean(df_hasta_2020['IAU_rend'])])
sigma1 = np.array([np.std(df_hasta_2020['IEF_rend']),np.std(df_hasta_2020['CETETRC.MX_rend']), np.std(df_hasta_2020['SPY_rend']), np.std(df_hasta_2020['EZA_rend']),np.std(df_hasta_2020['IAU_rend'])])
sesgo1  = np.array([skew(df_hasta_2020['IEF_rend']),skew(df_hasta_2020['CETETRC.MX_rend']), skew(df_hasta_2020['SPY_rend']), skew(df_hasta_2020['EZA_rend']),skew(df_hasta_2020['IAU_rend'])])
curtosis1 = np.array([kurtosis(df_hasta_2020['IEF_rend']),kurtosis(df_hasta_2020['CETETRC.MX_rend']), kurtosis(df_hasta_2020['SPY_rend']), kurtosis(df_hasta_2020['EZA_rend']),kurtosis(df_hasta_2020['IAU_rend'])])
matriz_Cov1 = df_hasta_2020[['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend']].cov().values
matriz_Corr1 = df_hasta_2020[['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend']].corr().values
#----------
mu2 = np.array([np.mean(df_desde_2020['IEF_rend']),np.mean(df_desde_2020['CETETRC.MX_rend']), np.mean(df_desde_2020['SPY_rend']), np.mean(df_desde_2020['EZA_rend']),np.mean(df_desde_2020['IAU_rend'])])
sigma2 = np.array([np.std(df_desde_2020['IEF_rend']),np.std(df_desde_2020['CETETRC.MX_rend']), np.std(df_desde_2020['SPY_rend']), np.std(df_desde_2020['EZA_rend']),np.std(df_desde_2020['IAU_rend'])])
sesgo2  = np.array([skew(df_desde_2020['IEF_rend']),skew(df_desde_2020['CETETRC.MX_rend']), skew(df_desde_2020['SPY_rend']), skew(df_desde_2020['EZA_rend']),skew(df_desde_2020['IAU_rend'])])
curtosis2 = np.array([kurtosis(df_desde_2020['IEF_rend']),kurtosis(df_desde_2020['CETETRC.MX_rend']), kurtosis(df_desde_2020['SPY_rend']), kurtosis(df_desde_2020['EZA_rend']),kurtosis(df_desde_2020['IAU_rend'])])
matriz_Cov2 = df_desde_2020[['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend']].cov().values
matriz_Corr2 = df_desde_2020[['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend']].corr().values
# Función para calcular el Sharpe Ratio
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

#VaR & CVaR
def calcular_var_cvar(returns, confidence=0.95):
    VaR = returns.quantile(1 - confidence)
    CVaR = returns[returns <= VaR].mean()
    return VaR, CVaR

def crear_histograma_distribucion(returns, var_95, cvar_95, title):
    # Crear el histograma base
    fig = go.Figure()
    
    # Calcular los bins para el histograma
    counts, bins = np.histogram(returns, bins=50)
    
    # Separar los bins en dos grupos: antes y después del VaR
    mask_before_var = bins[:-1] <= var_95
    
    # Añadir histograma para valores antes del VaR (rojo)
    fig.add_trace(go.Bar(
        x=bins[:-1][mask_before_var],
        y=counts[mask_before_var],
        width=np.diff(bins)[mask_before_var],
        name='Retornos < VaR',
        marker_color='rgba(255, 65, 54, 0.6)'
    ))
    
    # Añadir histograma para valores después del VaR (azul)
    fig.add_trace(go.Bar(
        x=bins[:-1][~mask_before_var],
        y=counts[~mask_before_var],
        width=np.diff(bins)[~mask_before_var],
        name='Retornos > VaR',
        marker_color='rgba(31, 119, 180, 0.6)'
    ))
    
    # Añadir líneas verticales para VaR y CVaR
    fig.add_trace(go.Scatter(
        x=[var_95, var_95],
        y=[0, max(counts)],
        mode='lines',
        name='VaR 95%',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=[cvar_95, cvar_95],
        y=[0, max(counts)],
        mode='lines',
        name='CVaR 95%',
        line=dict(color='purple', width=2, dash='dot')
    ))
    
    # Actualizar el diseño
    fig.update_layout(
        title=title,
        xaxis_title='Retornos',
        yaxis_title='Frecuencia',
        showlegend=True,
        barmode='overlay',
        bargap=0
    )
    
    return fig


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
    drawdown = (high_water_mark-precios) / high_water_mark
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
            x=precios.index,
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
    #st.write(datos)

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

#SOLUCIONAR PROBLEMA CON OTRO -----------------------------------------------------------------------------------------------------------------------------------
def drawdown2(simbolo=None, start_date=None, end_date=None, dataframe=None):
    """
    Realiza el análisis de drawdown.
    Si se proporciona un DataFrame, se usa directamente; de lo contrario, se descargan datos.
    """
    # Verificar si se proporciona un DataFrame o descargar datos
    if dataframe is not None:
        precios = dataframe
    else:
        if not simbolo:
            raise ValueError("Debe proporcionar un símbolo si no se pasa un DataFrame.")
        precios = yf.download(simbolo, start=start_date, end=end_date)['Close'].ffill().dropna()


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
def grafica_portafolio(df,w,columnas_rendimientos):
    df['Rend_Portafolio'] = df[columnas_rendimientos].dot(w)

# Crear una gráfica interactiva con Plotly
    fig = px.line(
    df,
    x='Date',
    y='Rend_Portafolio',
    title="Rendimientos del Portafolio",
    labels={'Date': 'Fecha', 'Rend_Portafolio': 'Rendimiento del Portafolio'},
    template="plotly_white"
    )

# Personalizar el diseño
    fig.update_layout(
    title_font=dict(size=22, family='Arial', color='darkblue'),
    xaxis_title="Fecha",
    yaxis_title="Rendimiento del Portafolio",
    xaxis=dict(showgrid=True, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridcolor='lightgray'),
    hovermode="x unified"
    )

# Mostrar la gráfica en Streamlit
    st.plotly_chart(fig, use_container_width=True)
  
def portafolio_estadistica(df,w,columnas_rendimientos):
    df['Rend_Portafolio'] = df[columnas_rendimientos].dot(w)
    s = estadisticas(df['Rend_Portafolio'])
    return s
def portfolio_stats(weights):
        columnas_rendimientos = ['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend']
        weights = np.array(weights)[:, np.newaxis]
        port_rets = weights.T @ np.array(df_hasta_2020[columnas_rendimientos].mean() * 252)[:, np.newaxis]
        port_vols = np.sqrt(np.dot(np.dot(weights.T, df_hasta_2020[columnas_rendimientos].cov() * 252), weights))
        return np.array([port_rets, port_vols, port_rets / port_vols]).flatten()

def min_sharpe_ratio(weights):
    return -portfolio_stats(weights)[2]


def grafica_portafolio_vs_emisora(df, w, columnas_rendimientos, emisora):
    
    # Calcular los rendimientos del portafolio
    df['Rend_Portafolio'] = df[columnas_rendimientos].dot(w)
    fig = px.line(
        df,
        x='Date',
        title="Comparación de Rendimientos: Portafolio vs. Emisora",
        labels={'Date': 'Fecha', 'value': 'Rendimiento'},
        template="plotly_white"
    )
    fig.add_scatter(
        x=df['Date'],
        y=df['Rend_Portafolio'],
        mode='lines',
        name='Rendimiento del Portafolio',
        line=dict(color='blue')
    )
    fig.add_scatter(
        x=df['Date'],
        y=df[emisora],
        mode='lines',
        name=f'Rendimiento de {emisora}',
        line=dict(color='orange')
    )
    fig.update_layout(
        title_font=dict(size=22, family='Arial', color='darkblue'),
        xaxis_title="Fecha",
        yaxis_title="Rendimientos",
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
  
def comparar_stats(v1, v2, v3, x, portafolio, ttl, etiquetas=None):
    # Crear un DataFrame para organizar los datos
    indices = range(len(v1)) if etiquetas is None else etiquetas
    data = pd.DataFrame({
        "Índice": indices,
        portafolio: v1,
        "S&P": v2,
        "Portafolio 1/5": v3
    })

    # Convertir a formato largo para facilitar la visualización con Plotly
    data_long = data.melt(id_vars="Índice", var_name="Instrumentos", value_name="Valor")

    # Crear el gráfico de barras
    fig = px.bar(
        data_long,
        x="Índice",
        y="Valor",
        color="Instrumentos",
        barmode="group",
        title= '             ' + ttl ,
        labels={"Índice": "_", "Valor": "Valor"}
    )

    # Personalizar el diseño
    fig.update_layout(
        title_font=dict(size=22, family='Arial', color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        hovermode="x unified",
        bargap=0.5,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white")
    )

    # Mostrar la gráfica
    st.plotly_chart(fig, use_container_width=True)

def drawdown3(dataframe):
    # Verificar si el DataFrame está vacío
    if dataframe.empty:
        st.error("El DataFrame proporcionado está vacío.")
        return

    # Verificar si las columnas requeridas están presentes
    columnas_requeridas = ['Date', 'Rend_Portafolio']
    for columna in columnas_requeridas:
        if columna not in dataframe.columns:
            st.error(f"Falta la columna requerida: {columna}")
            return

    # Asegurarse de que la columna 'Date' sea un índice de tipo datetime
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])  # Convertir la columna a datetime si no lo es
    dataframe.set_index('Date', inplace=True)

    precios = dataframe['Rend_Portafolio']  # Usar la columna de precios
    fig = graficar_drawdown_financiero(precios, f'Análisis de Drawdown')

    st.plotly_chart(fig)

    # Obtener información detallada del drawdown
    info_dd = obtener_max_drawdown_info(precios)

    # Imprimir resultados
    st.text(f"Máximo Drawdown: {info_dd['max_drawdown']:.2f}%")
    st.text(f"Fecha del pico: {info_dd['fecha_pico'].strftime('%Y-%m-%d')}")
    st.text(f"Fecha del valle: {info_dd['fecha_valle'].strftime('%Y-%m-%d')}")
    st.text(f"Duración de la caída: {info_dd['duracion_caida']} días")

    if info_dd['fecha_recuperacion'] is not None:
        st.text(f"Fecha de recuperación: {info_dd['fecha_recuperacion'].strftime('%Y-%m-%d')}")
        st.text(f"Duración de la recuperación: {info_dd['duracion_recuperacion']} días")
        st.text(f"Duración total: {info_dd['duracion_total']} días")
    else:
        st.text("El activo aún no se ha recuperado del máximo drawdown.")
#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Barra de navegación
st.sidebar.title("Navegación")
pages = [ "Selección de Activos", "Estadística de Activos", "Portafolios óptimos", "Backtesting", "Black-Litterman"]
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
    var1, cvar1 = calcular_var_cvar(df['IEF'])

    col1, col2, col13 = st.columns([3,1])
    if activo_seleccionado == "Activo 1":
      with col1:
        st.write("### Gráfica de Métricas")
        fig = px.line(
        df,
        x='Date',
        y='IEF_rend',
        title="Rendimientos de la Acción",
        labels={'fecha': "Fecha", 'rendimientos': "Rendimiento"},
        template="plotly_white")
        fig.update_layout(
        title_font=dict(size=20, family='Arial', color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        hovermode="x unified")
        st.plotly_chart(fig)
    
      with col2:
        st.write("### Datos del Activo Seleccionado")
        subcol1, subcol2 = st.columns(2)
        with subcol1: 
          e=estadisticas(df['IEF_rend'])
          st.text('     Rendimiento')
          st.subheader(f'     {round(e[0]*100,4)} %')
          st.text('     Sharp ratio')
          st.subheader(f'     {round(e[2],4)}')
          st.text('     Sesgo')
          st.subheader(f'     {round(e[4],4)}')
          
        with subcol2:
          st.text('     Volatilidad')
          st.subheader(f'     {round(e[1]*100,4)}%')
          st.text('     Sortino')
          st.subheader(f'     {round(e[3],4)}')
          st.text('     Curtosis')
          st.subheader(f'     {round(e[5],4)}')
          
        #st.text(estadisticas(df['IEF_rend']))
      
       with col13:
          st.text('     VaR')
          st.subheader(f"{var1:.2%}")
          st.text('     CVaR')
          st.subheader(f"{cvar1:.2%}")
         
      simbolo = 'IEF'
      start_date = '2010-01-01'
      end_date = datetime.now()
      drawdown(simbolo, start_date,end_date)

    if activo_seleccionado == "Activo 2":
        with col1:
          st.write("### Gráfica de Métricas")
          fig = px.line(
          df,
          x='Date',
          y='CETETRC.MX_rend',
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
          subcol1, subcol2 = st.columns(2)
          with subcol1: 
            e=estadisticas(df['CETETRC.MX_rend'])
            st.text('     Rendimiento')
            st.subheader(f'     {round(e[0]*100,4)} %')
            st.text('     Sharp ratio')
            st.subheader(f'     {round(e[2],4)}')
            st.text('     Sesgo')
            st.subheader(f'     {round(e[4],4)}')
          with subcol2:
            st.text('     Volatilidad')
            st.subheader(f'     {round(e[1]*100,4)}%')
            st.text('     Sortino')
            st.subheader(f'     {round(e[3],4)}')
            st.text('     Curtosis')
            st.subheader(f'     {round(e[5],4)}')
            
        simbolo = 'CETETRC.MX'
        start_date = '2010-01-01'
        end_date = datetime.now()
        drawdown(simbolo, start_date,end_date)

    if activo_seleccionado == "Activo 3":
        with col1:
          st.write("### Gráfica de Métricas")
          fig = px.line(
          df,
          x='Date',
          y='SPY_rend',
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
          subcol1, subcol2 = st.columns(2)
          with subcol1: 
            e=estadisticas(df['SPY_rend'])
            st.text('     Rendimiento')
            st.subheader(f'     {round(e[0]*100,4)} %')
            st.text('     Sharp ratio')
            st.subheader(f'     {round(e[2],4)}')
            st.text('     Sesgo')
            st.subheader(f'     {round(e[4],4)}')
          with subcol2:
            st.text('     Volatilidad')
            st.subheader(f'     {round(e[1]*100,4)}%')
            st.text('     Sortino')
            st.subheader(f'     {round(e[3],4)}')
            st.text('     Curtosis')
            st.subheader(f'     {round(e[5],4)}')
            
        simbolo = 'SPY'
        start_date = '2010-01-01'
        end_date = datetime.now()
        drawdown(simbolo, start_date,end_date)

    if activo_seleccionado == "Activo 4":
        with col1:
          st.write("### Gráfica de Métricas")
          fig = px.line(
          df,
          x='Date',
          y='EZA_rend',
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
          subcol1, subcol2 = st.columns(2)
          with subcol1: 
            e=estadisticas(df['EZA_rend'])
            st.text('     Rendimiento')
            st.subheader(f'     {round(e[0]*100,4)} %')
            st.text('     Sharp ratio')
            st.subheader(f'     {round(e[2],4)}')
            st.text('     Sesgo')
            st.subheader(f'     {round(e[4],4)}')
          with subcol2:
            st.text('     Volatilidad')
            st.subheader(f'     {round(e[1]*100,4)}%')
            st.text('     Sortino')
            st.subheader(f'     {round(e[3],4)}')
            st.text('     Curtosis')
            st.subheader(f'     {round(e[5],4)}')
            
        simbolo = 'EZA'
        start_date = '2010-01-01'
        end_date = datetime.now()
        drawdown(simbolo, start_date,end_date)      
    if activo_seleccionado == "Activo 5":
        with col1:
          st.write("### Gráfica de Métricas")
          fig = px.line(
          df,
          x='Date',
          y='IAU_rend',
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
          subcol1, subcol2 = st.columns(2)
          with subcol1: 
            e=estadisticas(df['IAU_rend'])
            st.text('     Rendimiento')
            st.subheader(f'     {round(e[0]*100,4)} %')
            st.text('     Sharp ratio')
            st.subheader(f'     {round(e[2],4)}')
            st.text('     Sesgo')
            st.subheader(f'     {round(e[4],4)}')
          with subcol2:
            st.text('     Volatilidad')
            st.subheader(f'     {round(e[1]*100,4)}%')
            st.text('     Sortino')
            st.subheader(f'     {round(e[3],4)}')
            st.text('     Curtosis')
            st.subheader(f'     {round(e[5],4)}')
            
        simbolo = 'IAU'
        start_date = '2010-01-01'
        end_date = datetime.now()
        drawdown(simbolo, start_date,end_date)  
# Portafolios^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
elif selection == "Portafolios óptimos":
    st.title("Portafolios óptimos")
    portafolios = ["Portafolio con mínima volatilidad", "Portafolio máximo sharpe ratio", "Portafolio mínima volatilidad con objetivo de rendimiento de 10%"]
    portafolio_seleccionado = st.selectbox("Selecciona un portafolio:", portafolios)
    st.write(f"Mostrando información para: {portafolio_seleccionado}")
    st.write("Aquí se mostrará información detallada del portafolio seleccionado.")
    
    
    if portafolio_seleccionado == "Portafolio con mínima volatilidad":
        mv = minima_varianza(matriz_Cov1)
        coll,colll ,collll= st.columns(3)
        with coll:
          #['IEF','CETETRC.MX', 'SPY', 'EZA','IAU']
          st.subheader('IEF')
          st.header(f'{round(mv[0]*100,3)}%')
          st.subheader('CETETRC.MX')
          st.header(f'{round(mv[1]*100,3)}%')
          
        with colll:
          st.subheader('EZA')
          st.header(f'{round(mv[3]*100,3)}%')
          st.subheader('IAU')
          st.header(f'{round(mv[4]*100,3)}%')
        with collll:
          st.subheader('SPY')
          st.header(f'{round(mv[2]*100,3)}%')
          
        grafica_portafolio(df_hasta_2020,mv,['IEF_rend','CETETRC.MX_rend', 'SPY_rend', 'EZA_rend','IAU_rend'])
        
    
    elif portafolio_seleccionado == "Portafolio máximo sharpe ratio":
        
        
        

        bnds = tuple((0, 1) for x in range(5))
        cons = {'type': 'eq', 'fun': lambda x: sum(x) - 1}
        initial_wts = 5 * [1. / 5]
        opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)
        pesos_optimos = opt_sharpe['x']
        if opt_sharpe.success:
            #st.write(list(zip(['IEF', 'CETETRC.MX', 'SPY', 'EZA', 'IAU'], round(opt_sharpe['x']*100, 2))))
            coll,colll,collll = st.columns(3)
            with coll:
          #['IEF','CETETRC.MX', 'SPY', 'EZA','IAU']
              st.subheader('IEF')
              st.header(f'{round(pesos_optimos[0]*100,3)}%')
              st.subheader('CETETRC.MX')
              st.header(f'{round(pesos_optimos[1]*100,3)}%')
              
            with colll:
              st.subheader('EZA')
              st.header(f'{round(pesos_optimos[3]*100,3)}%')
              st.subheader('IAU')
              st.header(f'{round(pesos_optimos[4]*100,3)}%')     
              #st.write(pesos_optimos)
            
            with collll:
              st.subheader('SPY')
              st.header(f'{round(pesos_optimos[2]*100,3)}%')

            grafica_portafolio(df_hasta_2020,pesos_optimos,['IEF_rend','CETETRC.MX_rend', 'SPY_rend', 'EZA_rend','IAU_rend'])
        else:
            st.write("Error en la optimización")
        

    elif portafolio_seleccionado == "Portafolio mínima volatilidad con objetivo de rendimiento de 10%":
        #st.text('ses')
        l = lagrange(mu1, matriz_Cov1, 0.10)
        coll,colll,collll = st.columns(3)
        with coll:
          #['IEF','CETETRC.MX', 'SPY', 'EZA','IAU']
          st.subheader('IEF')
          st.header(f'{round(l[0]*100,3)}%')
          st.subheader('CETETRC.MX')
          st.header(f'{round(l[1]*100,3)}%')
          
        with colll:
          st.subheader('EZA')
          st.header(f'{round(l[3]*100,3)}%')
          st.subheader('IAU')
          st.header(f'{round(l[4]*100,3)}%')     
        #st.write(l)
        with collll:
          st.subheader('SPY')
          st.header(f'{round(l[2]*100,3)}%')

        grafica_portafolio(df_hasta_2020,l,['IEF_rend','CETETRC.MX_rend', 'SPY_rend', 'EZA_rend','IAU_rend'])
  

# BACKTESTING´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´
elif selection == "Backtesting":
    st.title("Backtesting")
    portafolios = ["Portafolio con mínima volatilidad", "Portafolio máximo sharpe ratio", "Portafolio mínima volatilidad con objetivo de rendimiento de 10%"]
    portafolio_seleccionado = st.selectbox("Selecciona un portafolio:", portafolios)
    st.write(f"Mostrando información para: {portafolio_seleccionado}")
    st.write("Aquí se mostrará información detallada del portafolio seleccionado.")
    
    
    if portafolio_seleccionado == "Portafolio con mínima volatilidad":
      mv = minima_varianza(matriz_Cov1)
      f= portafolio_estadistica(df_desde_2020,mv,['IEF_rend','CETETRC.MX_rend', 'SPY_rend', 'EZA_rend','IAU_rend'])
      subcol1, subcol2,subcol3 = st.columns(3)
      with subcol1: 
        e=estadisticas(df['CETETRC.MX_rend'])
        st.text('     Rendimiento')
        st.subheader(f'     {round(f[0]*100,4)} %')
        st.text('     Sharp ratio')
        st.subheader(f'     {round(f[2],4)}')
        
      with subcol2:
        st.text('     Volatilidad')
        st.subheader(f'     {round(f[1]*100,4)}%')
        st.text('     Sortino')
        st.subheader(f'     {round(f[3],4)}')
        
      with subcol3:
        st.text('     Sesgo')
        st.subheader(f'     {round(f[4],4)}')
        st.text('     Curtosis')
        st.subheader(f'     {round(f[5],4)}')
      
      st.write(f'{f}')
      columnas_rendimientos =  ['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend']
      df['Rend_Portafolio'] = df[columnas_rendimientos].dot(mv)
      col1,col2,col3 = st.columns(3)
      with col1:
        comparar_stats(f[0],estadisticas(df_desde_2020['^GSPC_rend'])[0],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[0]
                       ,['p','s&p','ew'],'Portafolio min vol','Rendimiento',['Rendimiento'])
        comparar_stats(f[3],estadisticas(df_desde_2020['^GSPC_rend'])[3],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[3]
                       ,['p','s&p','ew'],'Portafolio min vol','Sortino',['Sortino'])
      with col2:
        comparar_stats(f[1],estadisticas(df_desde_2020['^GSPC_rend'])[1],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[1]
                       ,['p','s&p','ew'],'Portafolio min vol','Volatilidad',['Volatilidad'])
        comparar_stats(f[4],estadisticas(df_desde_2020['^GSPC_rend'])[4],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[4]
                       ,['p','s&p','ew'],'Portafolio min vol','Sesgo',['Sesgo'])
      with col3:
        comparar_stats(f[2],estadisticas(df_desde_2020['^GSPC_rend'])[2],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[3]
                       ,['p','s&p','ew'],'Portafolio min vol','Sharp ratio',['Sharp ratio'])
        comparar_stats(f[5],estadisticas(df_desde_2020['^GSPC_rend'])[5],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[5]
                       ,['p','s&p','ew'],'Portafolio min vol','Curtosis',['Curtosis'])
      
          
     # st.write(df['Rend_Portafolio'])
      #simbolo = 'Rend_Portafolio'
      #start_date = '2020-01-01'
      #end_date = datetime.now()
      #drawdown2(simbolo,df_desde_2020[['Date','Rend_Portafolio']])
      drawdown3(df_desde_2020[['Date','Rend_Portafolio']])
      #simbolo = df['Rend_Portafolio'] 
      #start_date = '2020-01-01'
      #end_date = datetime.now()
      #drawdown2(simbolo, start_date,end_date)
        
    
    elif portafolio_seleccionado == "Portafolio máximo sharpe ratio":
      #columnas_rendimientos =  ['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend']
        
      bnds = tuple((0, 1) for x in range(5))
      cons = {'type': 'eq', 'fun': lambda x: sum(x) - 1}
      initial_wts = 5 * [1. / 5]
      opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)
      pesos_optimos = opt_sharpe['x']
      
      r= portafolio_estadistica(df_desde_2020,pesos_optimos,['IEF_rend','CETETRC.MX_rend', 'SPY_rend', 'EZA_rend','IAU_rend'])
      #st.write(f'{f}')
      subcol1, subcol2, subcol3 = st.columns(3)
      with subcol1: 
        e=estadisticas(df['CETETRC.MX_rend'])
        st.text('     Rendimiento')
        st.subheader(f'     {round(r[0]*100,4)} %')
        st.text('     Sharp ratio')
        st.subheader(f'     {round(r[2],4)}')
        
      with subcol2:
        st.text('     Volatilidad')
        st.subheader(f'     {round(r[1]*100,4)}%')
        st.text('     Sortino')
        st.subheader(f'     {round(r[3],4)}')
        
      with subcol3:
        st.text('     Curtosis')
        st.subheader(f'     {round(r[5],4)}')
        st.text('     Sesgo')
        st.subheader(f'     {round(r[4],4)}')

      
      st.text('f')
      columnas_rendimientos =  ['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend']
      df['Rend_Portafolio'] = df[columnas_rendimientos].dot(pesos_optimos)
      col1,col2,col3 = st.columns(3)
      with col1:
        comparar_stats(r[0],estadisticas(df_desde_2020['^GSPC_rend'])[0],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[0]
                       ,['p','s&p','ew'],'Portafolio sharp','Rendimiento',['Rendimiento'])
        comparar_stats(r[3],estadisticas(df_desde_2020['^GSPC_rend'])[3],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[3]
                       ,['p','s&p','ew'],'Portafolio sharp','Sortino',['Sortino'])
      with col2:
        comparar_stats(r[1],estadisticas(df_desde_2020['^GSPC_rend'])[1],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[1]
                       ,['p','s&p','ew'],'Portafolio sharp','Volatilidad',['Volatilidad'])
        comparar_stats(r[4],estadisticas(df_desde_2020['^GSPC_rend'])[4],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[4]
                       ,['p','s&p','ew'],'Portafolio sharp','Sesgo',['Sesgo'])
      with col3:
        comparar_stats(r[2],estadisticas(df_desde_2020['^GSPC_rend'])[2],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[3]
                       ,['p','s&p','ew'],'Portafolio sharp','Sharp ratio',['Sharp ratio'])
        comparar_stats(r[5],estadisticas(df_desde_2020['^GSPC_rend'])[5],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[5]
                       ,['p','s&p','ew'],'Portafolio sharp','Curtosis',['Curtosis'])
     # st.write(df['Rend_Portafolio'])
      #simbolo = 'Rend_Portafolio'
      #start_date = '2020-01-01'
      #nd_date = datetime.now()
      
      #drawdown2(simbolo,df_desde_2020[['Date','Rend_Portafolio']])
      #simbolo = df['Rend_Portafolio'] 
      #start_date = '2020-01-01'
      #end_date = datetime.now()
      
      drawdown3(df_desde_2020[['Date','Rend_Portafolio']])
        

    elif portafolio_seleccionado == "Portafolio mínima volatilidad con objetivo de rendimiento de 10%":
      l = lagrange(mu1, matriz_Cov1, 0.10)
      ll= portafolio_estadistica(df_desde_2020,l,['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])
      #st.write(f'{f}')
      subcol1, subcol2,subcol3 = st.columns(3)
      with subcol1: 
        e=estadisticas(df['CETETRC.MX_rend'])
        st.text('     Rendimiento')
        st.subheader(f'     {round(ll[0]*100,4)} %')
        st.text('     Sharp ratio')
        st.subheader(f'     {round(ll[2],4)}')
      with subcol2:
        st.text('     Volatilidad')
        st.subheader(f'     {round(ll[1]*100,4)}%')
        st.text('     Sortino')
        st.subheader(f'     {round(ll[3],4)}')
      with subcol3:
        st.text('     Sesgo')
        st.subheader(f'     {round(ll[4],4)}')
        st.text('     Curtosis')
        st.subheader(f'     {round(ll[5],4)}')

      
      st.text('f')
      
      columnas_rendimientos =  ['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend']
      df_desde_2020['Rend_Portafolio'] = df_desde_2020[columnas_rendimientos].dot(l)
      st.write(df)
      #grafica_portafolio_vs_emisora(df_desde_2020, l, columnas_rendimientos, '^GSPC_rend')
      col1,col2,col3 = st.columns(3)
      with col1:
        comparar_stats(ll[0],estadisticas(df_desde_2020['^GSPC_rend'])[0],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[0]
                       ,['p','s&p','ew'],'Portafolio 10%','Rendimiento',['Rendimiento'])
        comparar_stats(ll[3],estadisticas(df_desde_2020['^GSPC_rend'])[3],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[3]
                       ,['p','s&p','ew'],'Portafolio 10%','Sortino',['Sortino'])
      with col2:
        comparar_stats(ll[1],estadisticas(df_desde_2020['^GSPC_rend'])[1],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[1]
                       ,['p','s&p','ew'],'Portafolio 10%','Volatilidad',['Volatilidad'])
        comparar_stats(ll[4],estadisticas(df_desde_2020['^GSPC_rend'])[4],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[4]
                       ,['p','s&p','ew'],'Portafolio 10%','Sesgo',['Sesgo'])
      with col3:
        comparar_stats(ll[2],estadisticas(df_desde_2020['^GSPC_rend'])[2],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[3]
                       ,['p','s&p','ew'],'Portafolio 10%','Sharp ratio',['Sharp ratio'])
        comparar_stats(ll[5],estadisticas(df_desde_2020['^GSPC_rend'])[5],
                       portafolio_estadistica(df_desde_2020,[0.2,0.2,0.2,0.2,0.2],['IEF_rend','CETETRC.MX_rend','SPY_rend','EZA_rend','IAU_rend'])[5]
                       ,['p','s&p','ew'],'Portafolio 10%','Curtosis',['Curtosis'])
     # st.write(df['Rend_Portafolio'])
      simbolo = 'Rend_Portafolio'
      start_date = '2020-01-01'
      end_date = datetime.now()
      st.write(df_desde_2020)
      st.write(df_desde_2020[['Date','Rend_Portafolio']])
      #df_desde_2020.set_index("Date", inplace=True)
      drawdown3(df_desde_2020[['Date','Rend_Portafolio']])
      
# Black-Litterman
elif selection == "Black-Litterman":
    st.title("Modelo Black-Litterman")
    
    st.write("Información sobre el modelo Black-Litterman será presentada aquí.")
    st.title("Selección de Activos")
    st.write("views:")
    st.text_input("Activo 1:")
    st.text_input("Activo 2:")
    st.text_input("Activo 3:")
    st.text_input("Activo 4:")
    st.text_input("Activo 5:")
      
