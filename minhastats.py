import streamlit as st
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
    try:
        return pd.read_csv("uso_da_frota_2025.csv")
    except Exception:
        return None

df = carregar_dados()

if df is None:
    st.error("Arquivo 'uso_da_frota_2025.csv' não encontrado no repositório.")
else:
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    
    aba0, aba2, aba3_4, aba5 = st.tabs([
        "📦 Módulo 0: Dados", 
        "📊 Módulo 2: Descritiva", 
        "🎲 Módulos 3/4: Simulações", 
        "📈 Módulo 5: Regressão"
    ])

    # Módulo 0
    with aba0:
        st.subheader("Dataset Carregado")
        st.write(f"Total de registros: **{len(df)}**")
        st.dataframe(df.head(10))

    # Módulo 2
    with aba2:
        st.subheader("Análise Descritiva Interativa")
        var_sel = st.selectbox("Selecione uma variável numérica:", cols_num)
        dados = df[var_sel].dropna().tolist()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Média (Própria)", f"{calcular_media(dados):.2f}")
        c2.metric("Mediana (Própria)", f"{calcular_mediana(dados):.2f}")
        c3.metric("Desvio Padrão", f"{calcular_desvio_padrao(dados):.2f}")
        c4.metric("Coef. Variação", f"{calcular_cv(dados):.2f}%")
        
        q1, q3 = calcular_percentil(dados, 25), calcular_percentil(dados, 75)
        iqr = q3 - q1
        outliers = [x for x in dados if x < (q1 - 1.5 * iqr) or x > (q3 + 1.5 * iqr)]
        st.write(f"**Regra do IQR:** Q1={q1:.2f}, Q3={q3:.2f}, Outliers detectados: **{len(outliers)}**")
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(dados, bins=15, color='skyblue', edgecolor='black')
        ax[0].set_title("Histograma")
        ax[1].boxplot(dados, vert=False)
        ax[1].set_title("Boxplot")
        st.pyplot(fig)

    # Módulos 3 e 4
    with aba3_4:
        st.subheader("Teorema Central do Limite (Monte Carlo)")
        var_sim = st.selectbox("Variável para amostragem:", cols_num, key="sim")
        dados_sim = df[var_sim].dropna().values
        
        n_amostra = st.slider("Tamanho da amostra (n):", 5, 100, 30)
        n_repeticoes = st.slider("Número de repetições:", 100, 5000, 1000)
        
        medias_amostrais = [np.mean(np.random.choice(dados_sim, size=n_amostra)) for _ in range(n_repeticoes)]
        
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.hist(medias_amostrais, bins=30, density=True, alpha=0.6, color='g')
        ax.set_title(f"Distribuição das Médias Amostrais (n={n_amostra})")
        st.pyplot(fig)

    # Módulo 5
    with aba5:
        st.subheader("Correlação e Regressão Linear")
        col_x = st.selectbox("Variável X (Independente):", cols_num, index=0)
        col_y = st.selectbox("Variável Y (Dependente):", cols_num, index=min(1, len(cols_num)-1))
        
        sub_df = df[[col_x, col_y]].dropna()
        vx, vy = sub_df[col_x].tolist(), sub_df[col_y].tolist()
        
        r = calcular_pearson(vx, vy)
        a, b, r2 = calcular_regressao(vx, vy)
        
        st.write(f"**Coeficiente de Correlação (r):** {r:.4f}")
        st.write(f"**Equação da Reta:** Y = {a:.4f}X + ({b:.4f}) | **R²:** {r2:.4f}")
        st.info("⚠️ **Aviso:** Correlação não implica causalidade!")
        
        val_x = st.number_input(f"Digitar valor para {col_x} (Predição):", value=float(calcular_media(vx)))
        st.success(f"**Predição Ŷ:** {a * val_x + b:.4f}")
        
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(vx, vy, alpha=0.5)
        x_trend = np.linspace(min(vx), max(vx), 100)
        ax.plot(x_trend, a * x_trend + b, color='red')
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        st.pyplot(fig)
