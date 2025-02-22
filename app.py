"""
Módulo com funções de endpoint a serem servidas pela API Flask.
"""

import os
from datetime import datetime
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request

from smoke_detection.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

UPLOAD_FOLDER = "artifacts/predictions"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET"])
def homePage():
    """Home page a ser renderizada"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def index():
    """Endpoint de predição para verificação de fumaça"""
    try:
        # imagem_recebida

        print(request.files)
        if "file" not in request.files:
            return "Imagem não enviada.", 400

        file = request.files["file"]
        if file.filename == "":
            return "Nenhum arquivo selecionado.", 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            today_date = datetime.today().strftime("%Y-%m-%d")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            prediction_folder = os.path.join(
                app.config["UPLOAD_FOLDER"], today_date, timestamp
            )
            os.makedirs(prediction_folder, exist_ok=True)

            filepath = os.path.join(prediction_folder, filename)
            file.save(filepath)

            # Abrir a imagem para verificar (opcional)
            image = Image.open(filepath)
            image.verify()

            obj = PredictionPipeline()

            # TODO Aplicar pré-processamento

            results = obj.predict("test.jpg")

            predictions = []
            for result in results:
                for box in result.boxes:
                    predictions.append(
                        {
                            "xmin": float(box.xyxy[0][0]),
                            "ymin": float(box.xyxy[0][1]),
                            "xmax": float(box.xyxy[0][2]),
                            "ymax": float(box.xyxy[0][3]),
                            "confidence": float(box.conf[0]),
                            "class_id": int(box.cls[0]),
                        }
                    )

            return {"predictions": predictions}

        return "Arquivo não permitido.", 400

    except ValueError as e:
        print(f"Não foi possível: {e}")
        return "falha"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
    # app.run(host="0.0.0.0", port=8080)
