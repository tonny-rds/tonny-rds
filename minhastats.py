import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# MÓDULO 1: NÚCLEO ESTATÍSTICO PRÓPRIO ("NA UNHA")
# ==========================================

def calcular_media(lista):
    return sum(lista) / len(lista) if lista else 0.0

def calcular_mediana(lista):
    if not lista: return 0.0
    s = sorted(lista)
    n = len(s)
    m = n // 2
    return (s[m - 1] + s[m]) / 2.0 if n % 2 == 0 else float(s[m])

def calcular_moda(lista):
    if not lista: return "N/A"
    freq = {}
    for x in lista:
        freq[x] = freq.get(x, 0) + 1
    max_freq = max(freq.values())
    if max_freq == 1:
        return "Sem moda (Amodal)"
    modas = [k for k, v in freq.items() if v == max_freq]
    if len(modas) == len(freq):
        return "Sem moda (Amodal)"
    elif len(modas) == 1:
        val = modas[0]
        return f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
    else:
        return f"Multimodal ({len(modas)} modas)"

def calcular_amplitude(lista):
    if not lista: return 0.0
    return float(max(lista) - min(lista))

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
    
    # Cálculo do R²
    y_pred = [a * xi + b for xi in x]
    stt = sum((yi - my) ** 2 for yi in y)
    stre = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))
    r2 = 1.0 - (stre / stt) if stt != 0 else 0.0
    
    return a, b, r2

def interpretar_assimetria(media, mediana, desvio):
    if desvio == 0: return "Distribuição Constante"
    dif = media - mediana
    limiar = 0.05 * desvio
    if abs(dif) <= limiar:
        return "Aproximadamente Simétrica (Média ≈ Mediana)"
    elif dif > 0:
        return "Assimétrica Positiva / à Direita (Média > Mediana)"
    else:
        return "Assimétrica Negativa / à Esquerda (Média < Mediana)"

# ==========================================
# CONFIGURAÇÃO E LEITURA DE DADOS
# ==========================================

st.set_page_config(page_title="Laboratório Estatístico - Frota", layout="wide")
st.title("🧮 Laboratório Estatístico — Análise da Frota 2025")

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

if df is None:
    st.warning("⚠️ O arquivo 'uso_da_frota_2025.csv' não foi localizado automaticamente.")
    arquivo_enviado = st.file_uploader("Faça o upload do arquivo CSV aqui:", type=["csv"])
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
# ESTRUTURA DOS MÓDULOS
# ==========================================

if df is not None:
    cols_num = [c for c in df.select_dtypes(include=[np.number]).columns if df[c].dropna().count() > 0]
    cols_cat = df.select_dtypes(include=['object', 'category']).columns.tolist()

    aba0, aba2, aba3, aba4, aba5 = st.tabs([
        "📦 Módulo 0: Dados Reais", 
        "📊 Módulo 2: Estatística Descritiva", 
        "🎲 Módulo 3: Monte Carlo (LGN & TCL)", 
        "📐 Módulo 4: Distribuições Teóricas",
        "📈 Módulo 5: Correlação & Regressão"
    ])

    # ------------------------------------------
    # MÓDULO 0: DADOS REAIS
    # ------------------------------------------
    with aba0:
        st.subheader("Visão Geral do Dataset")
        st.markdown(f"**Total de Registros:** `{len(df)}` | **Total de Colunas:** `{len(df.columns)}`")
        st.markdown(f"**Variáveis Numéricas ({len(cols_num)}):** {', '.join(cols_num)}")
        st.markdown(f"**Variáveis Categóricas ({len(cols_cat)}):** {', '.join(cols_cat[:8])}...")
        st.dataframe(df.head(15), use_container_width=True)

    # ------------------------------------------
    # MÓDULO 2: ESTATÍSTICA DESCRITIVA INTERATIVA
    # ------------------------------------------
    with aba2:
        st.subheader("Análise Descritiva Interativa")
        tipo_var = st.radio("Selecione o tipo de variável para análise:", ["Numérica", "Categórica"], horizontal=True)

        if tipo_var == "Numérica" and cols_num:
            var_sel = st.selectbox("Selecione uma variável numérica:", cols_num)
            dados = df[var_sel].dropna().tolist()

            med_p = calcular_media(dados)
            mediana_p = calcular_mediana(dados)
            desvio_p = calcular_desvio_padrao(dados, amostral=True)
            desvio_pop = calcular_desvio_padrao(dados, amostral=False)
            var_p = calcular_variancia(dados, amostral=True)
            var_pop = calcular_variancia(dados, amostral=False)
            moda_p = calcular_moda(dados)
            amp_p = calcular_amplitude(dados)
            cv_p = calcular_cv(dados)

            st.markdown("### 🔢 Medidas de Tendência Central e Dispersão")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Média", f"{med_p:.2f}")
            c2.metric("Mediana", f"{mediana_p:.2f}")
            c3.metric("Moda", str(moda_p))
            c4.metric("Amplitude Total", f"{amp_p:.2f}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Desvio Padrão (Amostra)", f"{desvio_p:.2f}")
            c6.metric("Desvio Padrão (População)", f"{desvio_pop:.2f}")
            c7.metric("Variância (Amostra)", f"{var_p:.2f}")
            c8.metric("Coef. Variação (CV)", f"{cv_p:.2f}%")

            # Interpretação textual automática
            txt_assim = interpretar_assimetria(med_p, mediana_p, desvio_p)
            st.info(f"💡 **Interpretação Automática de Formato:** {txt_assim}")

            # Detecção de Outliers via IQR
            q1 = calcular_percentil(dados, 25)
            q3 = calcular_percentil(dados, 75)
            iqr = q3 - q1
            lim_inf = q1 - 1.5 * iqr
            lim_sup = q3 + 1.5 * iqr
            outliers = [x for x in dados if x < lim_inf or x > lim_sup]

            st.markdown("---")
            st.markdown(f"**Regra do IQR:** Q1 = `{q1:.2f}` | Q3 = `{q3:.2f}` | IQR = `{iqr:.2f}`")
            st.markdown(f"**Limites de Outliers:** `[{lim_inf:.2f}, {lim_sup:.2f}]` → **Outliers detectados:** `{len(outliers)}` registro(s)")

            col_g1, col_g2 = st.columns(2)
            fig1, ax1 = plt.subplots(figsize=(6, 3.5))
            ax1.hist(dados, bins=20, color='#4C72B0', edgecolor='black')
            ax1.set_title(f"Histograma de {var_sel}")
            col_g1.pyplot(fig1)

            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            ax2.boxplot(dados, vert=False)
            ax2.set_title(f"Boxplot de {var_sel}")
            col_g2.pyplot(fig2)

            # Tabela de Frequências com Classes
            st.markdown("### 📋 Tabela de Frequências por Classes")
            bins_count = 8
            counts, bin_edges = np.histogram(dados, bins=bins_count)
            tabela_freq = []
            n_tot = len(dados)
            cum_freq = 0
            for i in range(len(counts)):
                label_classe = f"[{bin_edges[i]:.2f} ├ {bin_edges[i+1]:.2f})"
                freq_abs = counts[i]
                freq_rel = (freq_abs / n_tot) * 100
                cum_freq += freq_abs
                tabela_freq.append({
                    "Classe": label_classe,
                    "Freq. Absoluta": freq_abs,
                    "Freq. Relativa (%)": f"{freq_rel:.2f}%",
                    "Freq. Acumulada": cum_freq
                })
            st.table(pd.DataFrame(tabela_freq))

        elif tipo_var == "Categórica" and cols_cat:
            var_cat_sel = st.selectbox("Selecione uma variável categórica:", cols_cat)
            serie_cat = df[var_cat_sel].astype(str).str.strip()
            tb_abs = serie_cat.value_counts().head(10)
            tb_rel = (serie_cat.value_counts(normalize=True).head(10) * 100).round(2)

            df_cat_summary = pd.DataFrame({
                "Freq. Absoluta": tb_abs,
                "Freq. Relativa (%)": tb_rel.astype(str) + "%"
            })

            st.markdown("### 📋 Tabela de Frequência (Top 10 Categorias)")
            st.table(df_cat_summary)

            cg1, cg2 = st.columns(2)
            fig_bar, ax_bar = plt.subplots(figsize=(6, 3.5))
            tb_abs.plot(kind='bar', ax=ax_bar, color='#55A868', edgecolor='black')
            ax_bar.set_title(f"Frequência por {var_cat_sel}")
            cg1.pyplot(fig_bar)

            fig_pie, ax_pie = plt.subplots(figsize=(6, 3.5))
            tb_abs.plot(kind='pie', ax=ax_pie, autopct='%1.1f%%', startangle=90)
            ax_pie.set_ylabel("")
            ax_pie.set_title(f"Distribuição Proporcional")
            cg2.pyplot(fig_pie)

    # ------------------------------------------
    # MÓDULO 3: PROBABILIDADE E SIMULAÇÃO (MONTE CARLO)
    # ------------------------------------------
    with aba3:
        st.subheader("Simulação de Monte Carlo")
        sim_choice = st.radio("Escolha o Experimento de Simulação:", 
                             ["(a) Lei dos Grandes Números (LGN)", "(b) Teorema Central do Limite (TCL)"], horizontal=True)

        if sim_choice.startswith("(a)"):
            st.markdown("### 🎲 (a) Lei dos Grandes Números (LGN)")
            st.write("Demonstração da convergência da frequência relativa do lançamento de um dado honesto de 6 lados em direção à probabilidade teórica (1/6 ≈ 0,1667).")
            
            n_lancamentos = st.slider("Número de lançamentos simulados:", min_value=100, max_value=20000, value=5000, step=500)
            face_alvo = st.selectbox("Face do dado a acompanhar:", [1, 2, 3, 4, 5, 6], index=5)
            
            lancamentos = np.random.randint(1, 7, size=n_lancamentos)
            ocorrencias = (lancamentos == face_alvo).astype(int)
            freq_acumulada = np.cumsum(ocorrencias) / np.arange(1, n_lancamentos + 1)
            
            fig_lgn, ax_lgn = plt.subplots(figsize=(9, 4))
            ax_lgn.plot(freq_acumulada, color='blue', label=f'Frequência Relativa da Face {face_alvo}')
            ax_lgn.axhline(y=1/6, color='red', linestyle='--', label='Probabilidade Teórica (1/6 = 0,1667)')
            ax_lgn.set_xlabel("Número de Lançamentos")
            ax_lgn.set_ylabel("Frequência Relativa Acumulada")
            ax_lgn.set_title("Convergência da Frequência Relativa (Monte Carlo)")
            ax_lgn.legend()
            st.pyplot(fig_lgn)
            
        else:
            st.markdown("### 📊 (b) Teorema Central do Limite (TCL)")
            st.write("Conforme o tamanho da amostra ($n$) cresce, a distribuição das médias amostrais se aproxima de uma Distribuição Normal.")
            
            var_tcl = st.selectbox("Selecione a variável do dataset:", cols_num, key="tcl_var")
            dados_tcl = df[var_tcl].dropna().values
            
            c_tcl1, c_tcl2 = st.columns(2)
            n_amostra = c_tcl1.slider("Tamanho da Amostra (n):", min_value=2, max_value=150, value=30, step=2)
            n_rep = c_tcl2.slider("Número de Repetições:", min_value=100, max_value=5000, value=1000, step=100)
            
            medias_amostrais = [np.mean(np.random.choice(dados_tcl, size=n_amostra, replace=True)) for _ in range(n_rep)]
            
            fig_tcl, ax_tcl = plt.subplots(figsize=(9, 4))
            count, bins, _ = ax_tcl.hist(medias_amostrais, bins=30, density=True, alpha=0.6, color='#4C72B0', edgecolor='black')
            
            mu_tcl = calcular_media(medias_amostrais)
            sigma_tcl = calcular_desvio_padrao(medias_amostrais)
            if sigma_tcl > 0:
                y_norm = (1 / (sigma_tcl * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((bins - mu_tcl) / sigma_tcl) ** 2)
                ax_tcl.plot(bins, y_norm, color='red', linewidth=2, label='Normal Teórica')
            
            ax_tcl.set_title(f"Distribuição das Médias Amostrais de {var_tcl} (n={n_amostra})")
            ax_tcl.legend()
            st.pyplot(fig_tcl)

    # ------------------------------------------
    # MÓDULO 4: DISTRIBUIÇÕES TEÓRICAS
    # ------------------------------------------
    with aba4:
        st.subheader("Ajuste de Distribuições Teóricas")
        var_dist = st.selectbox("Selecione a variável numérica:", cols_num, key="dist_var")
        dados_dist = np.array(df[var_dist].dropna().tolist())
        
        dist_tipo = st.selectbox("Escolha a distribuição teórica para sobrepor:", ["Normal", "Exponencial"])
        
        mu_hat = calcular_media(dados_dist)
        sigma_hat = calcular_desvio_padrao(dados_dist, amostral=True)
        
        fig_d, ax_d = plt.subplots(figsize=(9, 4))
        count_d, bins_d, _ = ax_d.hist(dados_dist, bins=25, density=True, alpha=0.5, color='gray', edgecolor='black', label='Dados Reais')
        
        if dist_tipo == "Normal" and sigma_hat > 0:
            y_teo = (1 / (sigma_hat * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((bins_d - mu_hat) / sigma_hat) ** 2)
            ax_d.plot(bins_d, y_teo, 'r-', linewidth=2, label=f'Normal Teórica (μ={mu_hat:.2f}, σ={sigma_hat:.2f})')
        elif dist_tipo == "Exponencial" and mu_hat > 0:
            lamb = 1.0 / mu_hat
            bins_pos = np.maximum(0, bins_d)
            y_teo = lamb * np.exp(-lamb * bins_pos)
            ax_d.plot(bins_d, y_teo, 'g-', linewidth=2, label=f'Exponencial Teórica (λ={lamb:.4f})')
            
        ax_d.set_title(f"Histograma de {var_dist} vs Ajuste Teórico ({dist_tipo})")
        ax_d.legend()
        st.pyplot(fig_d)
        st.caption("Discussão de ajuste: Verifique visualmente a aderência da curva vermelha/verde ao contorno das barras do histograma.")

    # ------------------------------------------
    # MÓDULO 5: CORRELAÇÃO E REGRESSÃO LINEAR
    # ------------------------------------------
    with aba5:
        st.subheader("Análise Bivariada — Correlação e Regressão Linear Simples")
        if len(cols_num) >= 2:
            cx1, cx2 = st.columns(2)
            col_x = cx1.selectbox("Variável X (Independente):", cols_num, index=0)
            col_y = cx2.selectbox("Variável Y (Dependente):", cols_num, index=min(1, len(cols_num) - 1))
            
            sub_df = df[[col_x, col_y]].dropna()
            vx = sub_df[col_x].tolist()
            vy = sub_df[col_y].tolist()
            
            if len(vx) > 1:
                r = calcular_pearson(vx, vy)
                a, b, r2 = calcular_regressao(vx, vy)
                
                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("Correlação de Pearson (r)", f"{r:.4f}")
                m2.metric("Inclinação (a)", f"{a:.4f}")
                m3.metric("Coeficiente R²", f"{r2:.4f}")
                
                st.write(f"**Equação da Reta:** `Y = {a:.4f} * X + ({b:.4f})`")
                st.warning("⚠️ **Alerta Metodológico Honestidade:** Correlação estatística **NÃO** implica relação de causalidade!")
                
                st.markdown("---")
                st.subheader("Predição Interativa (Ŷ)")
                val_x = st.number_input(f"Digite um valor para {col_x}:", value=float(calcular_media(vx)))
                pred_y = a * val_x + b
                st.success(f"**Valor estimado para {col_y} (Ŷ):** `{pred_y:.4f}`")
                
                fig_reg, ax_reg = plt.subplots(figsize=(8, 4))
                ax_reg.scatter(vx, vy, alpha=0.5, color='#1f77b4', label='Dados')
                min_x, max_x = min(vx), max(vx)
                if min_x == max_x: max_x += 1.0
                x_range = np.linspace(min_x, max_x, 100)
                ax_reg.plot(x_range, a * x_range + b, color='red', linewidth=2, label='Reta por Mínimos Quadrados')
                ax_reg.set_xlabel(col_x)
                ax_reg.set_ylabel(col_y)
                ax_reg.legend()
                st.pyplot(fig_reg)
            else:
                st.warning("Não há pares válidos compartilhados entre as variáveis.")
        else:
            st.error("São necessárias pelo menos duas colunas numéricas válidas.")

