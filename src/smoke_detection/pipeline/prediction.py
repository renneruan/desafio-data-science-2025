"""
Módulo do pipeline utilizado para predição.

Pensado inicialmente para processamento de apenas um dados recebido.

A partir do dado recebido carrega o modelo, aplica o pré-processamento
adequado aos dados de entrada e devolve a probabilidade da classe.
"""

import cv2
from ultralytics import YOLO
from smoke_detection.constants import MODEL_PATH

# from smoke_detection import logger


class PredictionPipeline:
    """
    Class contendo pipeline de predição, submete os dados de entrada ao mesmo
    pipeline de transformação ajustado na etapa de pré-processamento dos dados
    de treino.
    """

    def __init__(self):
        self.model = YOLO(MODEL_PATH)  # Carrega o modelo treinado

    def get_results_bounding_boxes(self, results):
        predictions = []
        for result in results:
            for box in result.boxes:
                box_cls = int(box.cls[0].item())
                box_conf = box.conf[0].item()

                predictions.append(
                    {
                        "xmin": float(box.xyxy[0][0]),
                        "ymin": float(box.xyxy[0][1]),
                        "xmax": float(box.xyxy[0][2]),
                        "ymax": float(box.xyxy[0][3]),
                        "label": f"{self.model.names[box_cls]} {box_conf:.2f}",
                        "confidence": float(box_conf),
                        "class_id": int(box_cls),
                    }
                )

        return predictions

    def create_image_with_bounding_box(self, image_path, predictions):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for prediction in predictions:

            xmin, ymin, xmax, ymax = map(
                int,
                [
                    prediction["xmin"],
                    prediction["ymin"],
                    prediction["xmax"],
                    prediction["ymax"],
                ],
            )

            img = cv2.rectangle(
                img,
                (xmin, ymin),
                (xmax, ymax),
                (0, 255, 0),
                2,
            )

            img = cv2.putText(
                img,
                prediction["label"],
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        output_path = image_path.replace(".jpg", "_output.jpg")

        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Imagem salva em: {output_path}")

        return output_path

    def predict(self, input_path):
        """
        Função que realiza a predição dos dados.

        Args:
            data (pd.DataFrame): Image de entrada para predição.
        """
        results = self.model(input_path)

        predictions = self.get_results_bounding_boxes(results)
        output_path = self.create_image_with_bounding_box(
            input_path, predictions
        )

        return {"predictions": predictions, "output_path": output_path}
