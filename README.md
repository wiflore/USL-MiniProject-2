# Clasificador de Textos por ODS

Clasificador multiclase que asigna textos en español a uno de los 16 Objetivos de Desarrollo Sostenible (ODS) usando TF-IDF + TruncatedSVD + LinearSVC.

## Estructura

- `Clasificador_ODS.ipynb` — Notebook con el pipeline completo
- `Clasificador_ODS.html` — Version HTML del notebook
- `app_streamlit.py` — Aplicacion Streamlit interactiva
- `*.joblib` — Artefactos del modelo serializado

## Ejecutar Streamlit localmente

```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

## Deploy en Railway

El proyecto incluye un `Dockerfile` listo para Railway. Simplemente conecta el repo y Railway lo detectara automaticamente.
