
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Cargar el archivo Excel
df = pd.read_excel("Requierimientos de Auditoria.xlsx", engine="openpyxl")

# Lista personalizada de stopwords en español
spanish_stopwords = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para",
    "con", "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este",
    "sí", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta",
    "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni",
    "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos",
    "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes",
    "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros",
    "mi", "mis", "tú", "te", "ti", "tu", "tus", "ellas", "nosotras", "vosotros", "vosotras",
    "os", "mío", "mía", "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos",
    "suyas", "nuestro", "nuestra", "nuestros", "nuestras", "vuestro", "vuestra", "vuestros",
    "vuestras", "esos", "esas", "estoy", "estás", "está", "estamos", "estáis", "están", "esté",
    "estés", "estemos", "estéis", "estén", "estaré", "estarás", "estará", "estaremos", "estaréis",
    "estarán", "estaría", "estarías", "estaríamos", "estaríais", "estarían", "estaba", "estabas",
    "estábamos", "estabais", "estaban", "estuve", "estuviste", "estuvo", "estuvimos", "estuvisteis",
    "estuvieron", "estuviera", "estuvieras", "estuviéramos", "estuvierais", "estuvieran", "estuviese",
    "estuvieses", "estuviésemos", "estuvieseis", "estuviesen", "estando", "estado", "estada", "estados",
    "estadas", "estad"
]

# Crear el vectorizador TF-IDF
vectorizer = TfidfVectorizer(stop_words=spanish_stopwords)
tfidf_matrix = vectorizer.fit_transform(df["Requerimiento"])

# Función para buscar auditoría por requerimiento en lenguaje natural
def buscar_auditoria(pregunta):
    pregunta_vector = vectorizer.transform([pregunta])
    similitudes = cosine_similarity(pregunta_vector, tfidf_matrix)
    indice_max = similitudes.argmax()
    similitud_max = similitudes[0, indice_max]
    if similitud_max < 0.2:
        return None, None, similitud_max
    auditoria = df.iloc[indice_max]["Auditoría"]
    requerimiento = df.iloc[indice_max]["Requerimiento"]
    return auditoria, requerimiento, similitud_max

# Interfaz Streamlit
st.set_page_config(page_title="Buscador de Auditorías", layout="centered")
st.title("🔍 Buscador de Auditorías por Requerimiento")
st.write("Ingresá una consulta en lenguaje natural para encontrar la auditoría correspondiente.")

consulta = st.text_input("¿Qué requerimiento estás buscando?")
if consulta:
    auditoria, requerimiento, score = buscar_auditoria(consulta)
    if auditoria:
        st.success(f"**Auditoría encontrada:** {auditoria}")
        st.write(f"**Requerimiento asociado:** {requerimiento}")
        st.caption(f"🔎 Similitud: {score:.2f}")
    else:
        st.warning("No se encontró una auditoría relevante para tu consulta. Probá con otras palabras.")
