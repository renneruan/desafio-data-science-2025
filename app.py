"""
Módulo com funções de endpoint a serem servidas pela API Flask.
"""

from flask import Flask, render_template

from smoke_detection.pipeline.prediction import PredictionPipeline

app = Flask(__name__)


@app.route("/", methods=["GET"])
def homePage():
    """Home page a ser renderizada"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def index():
    """Endpoint de predição para verificação de fumaça"""
    try:
        # imagem_recebida

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

    except ValueError as e:
        print(f"Não foi possível: {e}")
        return "falha"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
    # app.run(host="0.0.0.0", port=8080)
