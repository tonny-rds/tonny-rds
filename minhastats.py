import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# MÓDULO 1: NÚCLEO ESTATÍSTICO PRÓPRIO
# ==========================================

def calcular_media(lista):
    return sum(lista) / len(lista) if lista else 0

def calcular_mediana(lista):
    if not lista: return 0
    s = sorted(lista)
    n = len(s)
    m = n // 2
    return (s[m - 1] + s[m]) / 2 if n % 2 == 0 else s[m]

def calcular_variancia(lista, amostral=True):
    n = len(lista)
    if n <= 1: return 0
    m = calcular_media(lista)
    div = (n - 1) if amostral else n
    return sum((x - m) ** 2 for x in lista) / div

def calcular_desvio_padrao(lista, amostral=True):
    return calcular_variancia(lista, amostral) ** 0.5

def calcular_cv(lista):
    m = calcular_media(lista)
    return (calcular_desvio_padrao(lista) / m * 100) if m != 0 else 0

def calcular_percentil(lista, p):
    if not lista: return 0
    s = sorted(lista)
    n = len(s)
    k = (p / 100) * (n - 1)
    f = int(k)
    c = k - f
    return s[f] + c * (s[f + 1] - s[f]) if f + 1 < n else s[f]

def calcular_covariancia(x, y):
    n = len(x)
    if n <= 1 or len(y) != n: return 0
    mx, my = calcular_media(x), calcular_media(y)
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1)

def calcular_pearson(x, y):
    dpx, dpy = calcular_desvio_padrao(x), calcular_desvio_padrao(y)
    return calcular_covariancia(x, y) / (dpx * dpy) if dpx * dpy != 0 else 0

def calcular_regressao(x, y):
    n = len(x)
    mx, my = calcular_media(x), calcular_media(y)
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = sum((x[i] - mx) ** 2 for i in range(n))
    a = num / den if den != 0 else 0
    b = my - a * mx
    
    # Cálculo do R²
    y_pred = [a * xi + b for xi in x]
    stt = sum((yi - my) ** 2 for yi in y)
    stre = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))
    r2 = 1 - (stre / stt) if stt != 0 else 0
    
    return a, b, r2

# ==========================================
# INTERFACE STREAMLIT E MÓDULOS VISUAIS
# ==========================================

st.set_page_config(page_title="Laboratório Estatístico - Frota", layout="wide")
st.title("🧮 Laboratório Estatístico - Uso da Frota")

@st.cache_data
def carregar_dados():
    caminho_absoluto = os.path.join(os.path.dirname(__file__), "uso_da_frota_2025.csv")
    if os.path.exists(caminho_absoluto):
        return pd.read_csv(caminho_absoluto)
    try:
        return pd.read_csv("uso_da_frota_2025.csv")
    except Exception:
        return None

df = carregar_dados()

if df is None:
    st.warning("⚠️ O arquivo 'uso_da_frota_2025.csv' não foi localizado automaticamente no servidor.")
    arquivo_enviado = st.file_uploader("Faça o upload do arquivo 'uso_da_frota_2025.csv' aqui para abrir a aplicação:", type=["csv"])
    if arquivo_enviado is not None:
        df = pd.read_csv(arquivo_enviado)
