from flask import Flask, render_template, send_from_directory
from flask_cors import CORS

from api.routes import create_api
from config.settings import settings
from services.ai_analyst import AIAnalystService
from services.inference import InferenceService
from services.logging_store import init_store
from services import logging_store
from services.model_loader import load_inference_assets


def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": settings.cors_origins}})

    assets = load_inference_assets()
    inference_service = InferenceService(assets)
    ai_analyst_service = AIAnalystService()
    init_store()
    app.register_blueprint(create_api(inference_service, logging_store, ai_analyst_service), url_prefix="/api")

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/samples/<path:filename>")
    def serve_samples(filename):
        return send_from_directory("samples", filename)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=settings.debug)
