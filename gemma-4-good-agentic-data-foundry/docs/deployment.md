# Deployment

## Recommended path

This is a Streamlit app, so GitHub Pages is not enough. GitHub Pages hosts static HTML, CSS, and JavaScript, but this project needs a Python process.

Recommended public sharing flow:

1. Push this repository to GitHub.
2. Deploy it on Streamlit Community Cloud.
3. Share the generated `streamlit.app` URL.

## Streamlit Community Cloud settings

- Repository: your GitHub repo
- Branch: `main`
- Main file path: `app/streamlit_app.py`
- Python dependencies: `requirements.txt`

## Local model limitation

The local Ollama/Gemma path works on your machine because `gemma3n:e4b` is installed locally. A standard Streamlit Community Cloud app will not have your local Ollama model.

For public sharing, use one of these modes:

- Demo mode: leave local model generation unchecked. The deterministic trusted-query flow still works.
- Full Gemma mode: deploy on a machine/container that can run Gemma, or replace Ollama with a hosted Gemma-compatible API.

## Pre-deploy checks

```bash
python3 scripts/smoke_test.py
python3 scripts/check_local_model.py
streamlit run app/streamlit_app.py
```

