"""
Módulo do pipeline utilizado para predição.

Pensado inicialmente para processamento de apenas um dados recebido.

A partir do dado recebido carrega o modelo, aplica o pré-processamento
adequado aos dados de entrada e devolve a probabilidade da classe.
"""

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

    def predict(self, input_data):
        """
        Função que realiza a predição dos dados.

        Args:
            data (pd.DataFrame): Image de entrada para predição.
        """
        results = self.model(input_data)

        return results
