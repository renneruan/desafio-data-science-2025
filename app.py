"""
Módulo com funções de endpoint a serem servidas pela API Flask.
"""

import os
import random
from datetime import datetime

from PIL import Image
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from flask import Flask, jsonify, render_template, request, send_from_directory

from smoke_detection.pipeline.prediction import PredictionPipeline
from smoke_detection import logger

app = Flask(__name__)

# Constantes com caminhos para as pastas de imagens
TEST_IMAGES_FOLDER = "datasets/data/test/images/"
ARTIFACTS_FOLDER = "artifacts/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

app.config["ARTIFACTS_FOLDER"] = ARTIFACTS_FOLDER


def allowed_file(filename):
    """Verifica se a extensão do arquivo é permitida"""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def create_file_on_prediction_folder(file):
    """Cria o arquivo de imagem na pasta de predições"""

    filename = secure_filename(file.filename)
    today_date = datetime.today().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    prediction_folder = os.path.join(
        app.config["ARTIFACTS_FOLDER"],
        "predictions/",
        today_date,
        timestamp,
    )
    os.makedirs(prediction_folder, exist_ok=True)

    filepath = os.path.join(prediction_folder, filename)
    file.save(filepath)

    return filepath


@app.route("/", methods=["GET"])
def homePage():
    """Home page a ser renderizada"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def index():
    """Endpoint de predição para verificação de fumaça"""
    try:
        # imagem_recebida
        if "file" not in request.files:
            return "Imagem não enviada.", 400

        file = request.files["file"]
        if file.filename == "":
            return "Nenhum arquivo selecionado.", 400

        if file and allowed_file(file.filename):
            filepath = create_file_on_prediction_folder(file)

            # Abrir a imagem para verificar (opcional)
            image = Image.open(filepath)
            image.verify()

            obj = PredictionPipeline()

            # TODO Aplicar pré-processamento

            results = obj.predict(filepath)

            return jsonify(results)

        return "Arquivo não permitido.", 400

    except ValueError as e:
        print(f"Não foi possível: {e}")
        return "falha"


@app.route("/predict_multiple", methods=["GET"])
def get_random_predictions():
    """Endpoint de predição de múltiplas imagens e aleatórias"""
    try:
        image_files = [
            f for f in os.listdir(TEST_IMAGES_FOLDER) if f.endswith((".jpg"))
        ]
        random_images = random.sample(image_files, 5)
        logger.info(random_images)

        all_results = []
        for image_name in random_images:
            # Abrir a imagem para verificar (opcional)
            image_path = os.path.join(TEST_IMAGES_FOLDER, image_name)

            with open(image_path, "rb") as file:
                file_obj = FileStorage(
                    file, filename=image_name
                )  # Criar um objeto similar ao de upload
                filepath = create_file_on_prediction_folder(file_obj)

            image = Image.open(filepath)
            image.verify()

            obj = PredictionPipeline()

            # TODO Aplicar pré-processamento

            results = obj.predict(filepath)

            all_results.append(results)

        print(all_results)
        return jsonify(all_results)

    except ValueError as e:
        print(f"Não foi possível: {e}")
        return "falha"


@app.route("/artifacts/<path:filename>")
def serve_image(filename):
    return send_from_directory(app.config["ARTIFACTS_FOLDER"], filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
    # app.run(host="0.0.0.0", port=8080)
