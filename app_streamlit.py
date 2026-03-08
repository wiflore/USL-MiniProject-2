"""Aplicacion Streamlit para clasificacion de textos por ODS."""
import streamlit as st
import joblib

# Cargar artefactos del pipeline (mismo pipeline del notebook)
tfidf = joblib.load('tfidf_vectorizer.joblib')
svd = joblib.load('svd_reducer.joblib')
normalizer = joblib.load('normalizer.joblib')
model = joblib.load('linearsvc_model.joblib')
ods_nombres = joblib.load('ods_nombres.joblib')

st.set_page_config(page_title="Clasificador ODS", layout="centered")

st.title("Clasificador de Textos por ODS")
st.write("Ingrese un texto en espanol y el modelo predecira a cual de los 16 "
         "Objetivos de Desarrollo Sostenible (ODS) esta relacionado.")

texto_input = st.text_area(
    "Ingrese el texto a clasificar:",
    height=200,
    placeholder="Escriba o pegue aqui el texto que desea clasificar..."
)

if st.button("Clasificar"):
    if texto_input.strip():
        X_tfidf = tfidf.transform([texto_input])
        X_svd = normalizer.transform(svd.transform(X_tfidf))
        prediccion = model.predict(X_svd)[0]

        nombre_ods = ods_nombres.get(prediccion, "Desconocido")

        st.write(f"**Resultado:** ODS {prediccion} - {nombre_ods}")
    else:
        st.write("Por favor ingrese un texto para clasificar.")

st.write("Modelo entrenado con OSDG Community Dataset v2023")
