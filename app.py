"""
Simple wrapper to import the Flask app for Render.com deployment
This allows gunicorn to find the app as 'app:app'
"""
from backend_improved import app

if __name__ == '__main__':
    app.run()
