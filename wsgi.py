"""
wsgi.py
=======
WSGI entrypoint for production deployment.

The development server (`python app.py`) is fine for local demos but is
single-threaded and not intended for real traffic. A WSGI server imports the
`app` object from here instead:

    gunicorn --workers 4 --bind 0.0.0.0:8000 wsgi:app

Notes
-----
* `app` is imported from app.py, where the trained model, the classifier and
  the dataset are loaded once at import time. Because WSGI servers preload the
  application per worker, the artifacts are loaded once per worker process
  rather than once per request.
* The model artifacts must exist before the first request. If they are missing,
  the import fails loudly rather than serving wrong numbers.
"""

from app import app  # noqa: F401  (imported for the WSGI server)

if __name__ == "__main__":
    # Convenience for `python wsgi.py` — same behaviour as `python app.py`.
    app.run(host="127.0.0.1", port=5000)
