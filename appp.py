import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta


# Configuración inicial de la app
st.set_page_config(page_title="Aplicación de Portafolios", layout="wide")

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
    st.write(f"Mostrando estadísticas para: {activo_seleccionado}")
    st.write("Aquí se mostrarán las estadísticas relevantes del activo seleccionado.")

# Portafolio 1
elif selection == "Portafolio 1":
    st.title("Portafolio 1")
    portafolios = ["Portafolio A", "Portafolio B", "Portafolio C"]
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
