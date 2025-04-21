# declass-backend

This is the backend for the CADI declass application. All of the source code can be found in app.py, with a Dockerfile for building a Docker container and a requirements.txt file for dependencies.

## Setup

1. Install dependencies, in a Python 3.11 environment:
```bash
pip install -r requirements.txt
```

You will additionally need to install tesseract on your system. You can find the installation instructions on the [Tesseract Installation Page](https://tesseract-ocr.github.io/tessdoc/Installation.html).

You will also need to install [Poppler](https://poppler.freedesktop.org/).

2. Set up environment variables:
```bash
cp .env.example .env
```

You will need to set up the following environment variables:

- SUPABASE_URL
- SUPABASE_KEY
- SUPABASE_SERVICE_KEY
- ANTHROPIC_API_KEY
- GEMINI_API_KEY
- JWT_SECRET
- TESSERACT_PATH

The Supabase key, url, and service key can be found in the settings, under Configuration > Data API. The URL is listed at the top, the KEY is the anon public key, and the SERVICE_KEY is the service key (should have the secret tag next to it). The JWT_SECRET is also found here under JWT Settings.

An anthropic API key can be generated from the [Anthropic API Key Page](https://console.anthropic.com/settings/keys).

A Gemini API key can be generated from the [Gemini API Key Page](https://ai.google.dev/gemini-api/docs/api-key).

The TESSERACT_PATH should be the path to the tesseract executable on your system.

3. Run the application:
```bash
uvicorn app:app --reload
```

This will run the application locally at http://localhost:8000.

