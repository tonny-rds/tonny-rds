import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# MÓDULO 1: NÚCLEO ESTATÍSTICO PRÓPRIO
# ==========================================

def calcular_media(lista):
    return sum(lista) / len(lista) if lista else 0.0

def calcular_mediana(lista):
    if not lista: return 0.0
    s = sorted(lista)
    n = len(s)
    m = n // 2
    return (s[m - 1] + s[m]) / 2.0 if n % 2 == 0 else float(s[m])

def calcular_variancia(lista, amostral=True):
    n = len(lista)
    if n <= 1: return 0.0
    m = calcular_media(lista)
    div = (n - 1) if amostral else n
    return sum((x - m) ** 2 for x in lista) / div

def calcular_desvio_padrao(lista, amostral=True):
    return calcular_variancia(lista, amostral) ** 0.5

def calcular_cv(lista):
    m = calcular_media(lista)
    return (calcular_desvio_padrao(lista) / m * 100.0) if m != 0 else 0.0

def calcular_percentil(lista, p):
    if not lista: return 0.0
    s = sorted(lista)
    n = len(s)
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = k - f
    return s[f] + c * (s[f + 1] - s[f]) if f + 1 < n else float(s[f])

def calcular_covariancia(x, y):
    n = len(x)
    if n <= 1 or len(y) != n: return 0.0
    mx, my = calcular_media(x), calcular_media(y)
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1)

def calcular_pearson(x, y):
    dpx, dpy = calcular_desvio_padrao(x), calcular_desvio_padrao(y)
    return calcular_covariancia(x, y) / (dpx * dpy) if (dpx * dpy) != 0 else 0.0

def calcular_regressao(x, y):
    n = len(x)
    if n <= 1: return 0.0, 0.0, 0.0
    mx, my = calcular_media(x), calcular_media(y)
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = sum((x[i] - mx) ** 2 for i in range(n))
    a = num / den if den != 0 else 0.0
    b = my - a * mx
    
    # Cálculo do Coeficiente de Determinação (R²)
    y_pred = [a * xi + b for xi in x]
    stt = sum((yi - my) ** 2 for yi in y)
    stre = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))
    r2 = 1.0 - (stre / stt) if stt != 0 else 0.0
    
    return a, b, r2

# ==========================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ==========================================

st.set_page_config(page_title="Laboratório Estatístico - Frota", layout="wide")
st.title("🧮 Laboratório Estatístico - Uso da Frota")

# ==========================================
# FUNÇÃO ROBUSTA DE LEITURA DO CSV
# ==========================================

@st.cache_data
def carregar_dados():
    caminho_absoluto = os.path.join(os.path.dirname(__file__), "uso_da_frota_2025.csv")
    caminhos = [caminho_absoluto, "uso_da_frota_2025.csv"]
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    separadores = [',', ';']

    for caminho in caminhos:
        if os.path.exists(caminho):
            for enc in encodings:
                for sep in separadores:
                    try:
                        temp_df = pd.read_csv(caminho, encoding=enc, sep=sep)
                        if temp_df.shape[1] > 1:
                            return temp_df
                    except Exception:
                        continue
    return None

df = carregar_dados()

# Fallback para upload manual se o arquivo não for encontrado no servidor
if df is None:
    st.warning("⚠️ O arquivo 'uso_da_frota_2025.csv' não foi localizado automaticamente no servidor.")
    arquivo_enviado = st.file_uploader("Faça o upload manual do arquivo CSV aqui:", type=["csv"])
    if arquivo_enviado is not None:
        for enc in ['utf-8', 'latin1', 'cp1252']:
            for sep in [',', ';']:
                try:
                    arquivo_enviado.seek(0)
                    temp_df = pd.read_csv(arquivo_enviado, encoding=enc, sep=sep)
                    if temp_df.shape[1] > 1:
                        df = temp_df
                        break
                except Exception:
                    continue

# ==========================================
# EXIBIÇÃO DOS MÓDULOS DA APLICAÇÃO
# ==========================================

if df is not None:
    # Filtrar apenas colunas numéricas que tenham dados válidos (não 100% nulas)
    cols_num = [c for c in df.select_dtypes(include=[np.number]).columns if df[c].dropna().count() > 0]
    
    aba0, aba2, aba3_4, aba5 = st.tabs([
        "📦 Módulo 0: Dados", 
        "📊 Módulo 2: Descritiva", 
        "🎲 Módulos 3/4: Simulações", 
        "📈 Módulo 5: Regressão"
    ])

    # --- MÓDULO 0: DADOS REAIS ---
    with aba0:
        st.subheader("Visão Geral do Dataset")
        st.write(f"Total de registros: **{len(df)}** | Total de colunas: **{len(df.columns)}**")
        st.dataframe(df.head(15), use_container_width=True)

    # --- MÓDULO 2: ESTATÍSTICA DESCRITIVA INTERATIVA ---
    with aba2:
        st.subheader("Análise Descritiva Univariada")
        if cols_num:
            var_sel = st.selectbox("Selecione uma variável numérica:", cols_num)
            dados = df[var_sel].dropna().tolist()
            
            if len(dados) > 0:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Média (Própria)", f"{calcular_media(dados):.2f}")
                c2.metric("Mediana (Própria)", f"{calcular_mediana(dados):.2f}")
                c3.metric("Desvio Padrão (Amostral)", f"{calcular_desvio_padrao(dados):.2f}")
                c4.metric("Coeficiente de Variação", f"{calcular_cv(dados):.2f}%")
                
                q1 = calcular_percentil(dados, 25)
                q3 = calcular_percentil(dados, 75)
                iqr = q3 - q1
                outliers = [x for x in dados if x < (q1 - 1.5 * iqr) or x > (q3 + 1.5 * iqr)]
                
                st.markdown("---")
                st.write(f"**Regra do IQR:** Q1 = `{q1:.2f}` | Q3 = `{q3:.2f}` | IQR = `{iqr:.2f}`")
                st.write(f"**Outliers detectados:** **{len(outliers)}** elemento(s)")
                
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].hist(dados, bins=15, color='#4C72B0', edgecolor='black')
                ax[0].set_title(f"Histograma de {var_sel}")
                ax[1].boxplot(dados, vert=False)
                ax[1].set_title(f"Boxplot de {var_sel}")
                st.pyplot(fig)
            else:
                st.warning("Esta coluna não possui dados numéricos válidos.")
        else:
            st.error("Nenhuma coluna numérica válida identificada no arquivo.")

    # --- MÓDULOS 3 E 4: PROBABILIDADE, SIMULAÇÃO E DISTRIBUIÇÕES ---
    with aba3_4:
        st.subheader("Simulação de Monte Carlo — Teorema Central do Limite")
        if cols_num:
            var_sim = st.selectbox("Selecione a variável para amostragem:", cols_num, key="sim_var")
            dados_sim = df[var_sim].dropna().values
            
            if len(dados_sim) > 0:
                col_param1, col_param2 = st.columns(2)
                n_amostra = col_param1.slider("Tamanho da Amostra (n):", min_value=5, max_value=200, value=30, step=5)
                n_repeticoes = col_param2.slider("Número de Repetições:", min_value=100, max_value=5000, value=1000, step=100)
                
                medias_amostrais = [np.mean(np.random.choice(dados_sim, size=n_amostra, replace=True)) for _ in range(n_repeticoes)]
                
                fig, ax = plt.subplots(figsize=(8, 4))
                count, bins, ignored = ax.hist(medias_amostrais, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')
                
                # Ajuste da Curva Normal Teórica
                mu_sim = calcular_media(medias_amostrais)
                sigma_sim = calcular_desvio_padrao(medias_amostrais)
                if sigma_sim > 0:
                    y_normal = (1 / (sigma_sim * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((bins - mu_sim) / sigma_sim) ** 2)
                    ax.plot(bins, y_normal, linewidth=2, color='r', label='Ajuste Normal Teórico')
                    ax.legend()
                    
                ax.set_title(f"Distribuição das Médias Amostrais (n={n_amostra}, repetições={n_repeticoes})")
                st.pyplot(fig)
            else:
                st.warning("Esta coluna não possui dados válidos para simulação.")
        else:
            st.error("Nenhuma coluna numérica disponível para simulação.")

    # --- MÓDULO 5: CORRELAÇÃO E REGRESSÃO LINEAR ---
    with aba5:
        st.subheader("Análise Bivariada — Correlação e Regressão Linear Simples")
        if len(cols_num) >= 2:
            cx1, cx2 = st.columns(2)
            col_x = cx1.selectbox("Variável X (Independente):", cols_num, index=0)
            
            # Tentar selecionar como padrão a segunda coluna numérica da lista
            default_y_idx = 1 if len(cols_num) > 1 else 0
            col_y = cx2.selectbox("Variável Y (Dependente):", cols_num, index=default_y_idx)
            
            sub_df = df[[col_x, col_y]].dropna()
            vx = sub_df[col_x].tolist()
            vy = sub_df[col_y].tolist()
            
            if len(vx) > 1:
                r = calcular_pearson(vx, vy)
                a, b, r2 = calcular_regressao(vx, vy)
                
                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("Correlação de Pearson (r)", f"{r:.4f}")
                m2.metric("Inclinacao (a)", f"{a:.4f}")
                m3.metric("Coef. Determinação (R²)", f"{r2:.4f}")
                
                st.write(f"**Equação da Reta:** `Y = {a:.4f} * X + ({b:.4f})`")
                st.warning("⚠️ **Aviso Metodológico:** Correlação estatística não implica relação de causalidade!")
                
                st.markdown("---")
                st.subheader("Predição Interativa (Ŷ)")
                val_x = st.number_input(f"Digite um valor para {col_x}:", value=float(calcular_media(vx)))
                pred_y = a * val_x + b
                st.success(f"**Valor estimado para {col_y} (Ŷ):** `{pred_y:.4f}`")
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(vx, vy, alpha=0.5, color='#1f77b4', label='Dados')
                min_x, max_x = min(vx), max(vx)
                if min_x == max_x:
                    max_x += 1.0
                x_range = np.linspace(min_x, max_x, 100)
                ax.plot(x_range, a * x_range + b, color='red', linewidth=2, label='Reta de Regressão')
                ax.set_xlabel(col_x)
                ax.set_ylabel(col_y)
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("⚠️ Não há registros válidos compartilhados entre as duas variáveis selecionadas.")
        else:
            st.error("São necessárias pelo menos duas variáveis numéricas com dados válidos para a regressão.")

