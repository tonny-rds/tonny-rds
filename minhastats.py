import streamlit as st

# --- FUNÇÕES ESTATÍSTICAS ---

def calcular_media(lista):
    """Calcula a média aritmética."""
    if not lista:
        return 0
    return sum(lista) / len(lista)

def calcular_mediana(lista):
    """Calcula a mediana."""
    if not lista:
        return 0
    lista_ordenada = sorted(lista)
    n = len(lista_ordenada)
    meio = n // 2
    if n % 2 == 0:
        return (lista_ordenada[meio - 1] + lista_ordenada[meio]) / 2
    else:
        return lista_ordenada[meio]

def calcular_moda(lista):
    """Calcula a moda."""
    if not lista:
        return None
    frequencias = {}
    for valor in lista:
        frequencias[valor] = frequencias.get(valor, 0) + 1
    
    max_freq = max(frequencias.values())
    modas = [k for k, v in frequencias.items() if v == max_freq]
    
    if len(modas) == len(lista):
        return "Sem moda única"
    return modas if len(modas) > 1 else modas[0]

def calcular_amplitude(lista):
    """Calcula a amplitude total."""
    if not lista:
        return 0
    return max(lista) - min(lista)

def calcular_percentil(lista, p):
    """Calcula o percentil p (0 a 100)."""
    if not lista:
        return 0
    lista_ordenada = sorted(lista)
    n = len(lista_ordenada)
    if n == 1:
        return lista_ordenada[0]
    
    k = (p / 100) * (n - 1)
    f = int(k)
    c = k - f
    
    if f + 1 < n:
        return lista_ordenada[f] + c * (lista_ordenada[f + 1] - lista_ordenada[f])
    else:
        return lista_ordenada[f]

def calcular_quartis(lista):
    """Retorna Q1, Q2 (Mediana) e Q3."""
    q1 = calcular_percentil(lista, 25)
    q2 = calcular_percentil(lista, 50)
    q3 = calcular_percentil(lista, 75)
    return {"Q1": q1, "Q2": q2, "Q3": q3}

def calcular_variancia(lista, amostral=True):
    """Calcula a variância."""
    n = len(lista)
    if n <= 1:
        return 0
    media = calcular_media(lista)
    soma_quadrados = sum((x - media) ** 2 for x in lista)
    divisor = (n - 1) if amostral else n
    return soma_quadrados / divisor

def calcular_desvio_padrao(lista, amostral=True):
    """Calcula o desvio padrão."""
    var = calcular_variancia(lista, amostral)
    return var ** 0.5

def calcular_covariancia(x, y, amostral=True):
    """Calcula a covariância entre x e y."""
    n = len(x)
    if n <= 1 or len(y) != n:
        return 0
    media_x = calcular_media(x)
    media_y = calcular_media(y)
    
    soma_produtos = sum((x[i] - media_x) * (y[i] - media_y) for i in range(n))
    divisor = (n - 1) if amostral else n
    return soma_produtos / divisor

def calcular_correlacao_pearson(x, y):
    """Calcula o coeficiente de correlação de Pearson."""
    cov = calcular_covariancia(x, y, amostral=True)
    dp_x = calcular_desvio_padrao(x, amostral=True)
    dp_y = calcular_desvio_padrao(y, amostral=True)
    
    if dp_x == 0 or dp_y == 0:
        return 0.0
    return cov / (dp_x * dp_y)

def calcular_regressao_linear(x, y):
    """Calcula os coeficientes da reta de regressão linear."""
    n = len(x)
    if n == 0:
        return 0.0, 0.0
    media_x = calcular_media(x)
    media_y = calcular_media(y)
    
    numerador = sum((x[i] - media_x) * (y[i] - media_y) for i in range(n))
    denominador = sum((x[i] - media_x) ** 2 for i in range(n))
    
    if denominador == 0:
        return 0.0, media_y
    
    a = numerador / denominador
    b = media_y - a * media_x
    return a, b


# --- INTERFACE VISUAL STREAMLIT ---

st.set_page_config(page_title="Laboratório Estatístico", page_icon="📈")
st.title("📈 Laboratório Estatístico")
st.write("Insira seus dados abaixo para realizar a análise estatística.")

def processar_entrada(texto):
    try:
        return [float(x.strip()) for x in texto.split(",") if x.strip()]
    except ValueError:
        st.error("Erro de digitação: Insira apenas números separados por vírgula.")
        return []

st.subheader("Entrada de Dados")
entrada_x = st.text_input("Lista X (obrigatória - números separados por vírgula):", "10, 20, 30, 40, 50")
entrada_y = st.text_input("Lista Y (opcional - números separados por vírgula):", "12, 24, 33, 39, 55")

lista_x = processar_entrada(entrada_x)
lista_y = processar_entrada(entrada_y)

if st.button("Calcular Estatísticas"):
    if not lista_x:
        st.warning("Por favor, insira valores na Lista X.")
    else:
        st.divider()
        st.subheader("📊 Análise Univariada (Lista X)")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Média", f"{calcular_media(lista_x):.2f}")
        col2.metric("Mediana", f"{calcular_mediana(lista_x):.2f}")
        
        moda_result = calcular_moda(lista_x)
        if isinstance(moda_result, list):
             col3.metric("Moda", ", ".join([str(m) for m in moda_result]))
        else:
             col3.metric("Moda", str(moda_result))
             
        col4.metric("Amplitude", f"{calcular_amplitude(lista_x):.2f}")
        
        st.write(f"**Desvio Padrão (Amostral):** {calcular_desvio_padrao(lista_x):.4f}")
        st.write(f"**Variância (Amostral):** {calcular_variancia(lista_x):.4f}")
        
        quartis = calcular_quartis(lista_x)
        st.write(f"**Quartis:** Q1 = {quartis['Q1']:.2f} | Q2 = {quartis['Q2']:.2f} | Q3 = {quartis['Q3']:.2f}")

        if entrada_y:
            st.divider()
            st.subheader("📈 Análise Bivariada (X e Y)")
            if len(lista_x) != len(lista_y):
                st.error(f"As listas precisam ter o mesmo tamanho. X tem {len(lista_x)} itens e Y tem {len(lista_y)} itens.")
            else:
                st.write(f"**Covariância:** {calcular_covariancia(lista_x, lista_y):.4f}")
                st.write(f"**Correlação de Pearson (r):** {calcular_correlacao_pearson(lista_x, lista_y):.4f}")
                
                a, b = calcular_regressao_linear(lista_x, lista_y)
                st.write(f"**Reta de Regressão Linear:** Y = {a:.4f}X + {b:.4f}")
