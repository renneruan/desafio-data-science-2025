import os

# import cv2
import shutil
from ultralytics import YOLO

from smoke_detection import logger
from smoke_detection.constants import (
    DATASET_YAML_PATH,
    MODEL_PATH,
    OUTPUT_FOLDER,
)

from smoke_detection.utils.commons import get_latest_folder


all_datasets_path = DATASET_YAML_PATH


def move_best_model():
    latest_folder = get_latest_folder(OUTPUT_FOLDER)

    best_model_path = os.path.join(latest_folder, "weights/best.pt")

    logger.info("Melhor modelo acessado a partir da saída do treinamento.")
    logger.info(best_model_path)

    if os.path.exists(best_model_path):
        shutil.move(best_model_path, MODEL_PATH)
        logger.info(
            "Arquivo de melhor modelo movido com sucesso para a pasta %s.",
            MODEL_PATH,
        )
    else:
        logger.info(
            "O arquivo de melhor modelo não foi encontrado no caminho %s",
            best_model_path,
        )


def train(**args):
    model = YOLO("yolo11n.pt")  # pt para modelo pré-treinado

    # Treina o modelo no conjunto train da pasta data
    # imgsz já tem como valor padrão 640

    print(args)
    model.train(
        data=os.path.abspath(DATASET_YAML_PATH),
        epochs=1,
        batch=4,
        # device=0,
        seed=42,
    )

    logger.info("Treinamento finalizado.")
    return model


def evaluate_model(model):
    metrics = model.val()
    logger.info(metrics.results_dict)
    logger.info(metrics.speed)


def train_pipeline():
    model = train()
    move_best_model()
    evaluate_model(model)
