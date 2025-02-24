import cv2
import os
from src.smoke_detection.utils.commons import read_yaml
from src.smoke_detection import logger

from smoke_detection.constants import (
    DATASET_YAML_PATH,
    DATASET_PATH,
    PROCESSED_DATASET_PATH,
)


def equalize_brightness_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # CLAHE significa "Contrast Limited Adaptive Histogram Equalization"
    # Processa o contraste das imagens realizando uma equalização
    # Deixa áreas escuras mais claras e áreas muito claras mais escuras
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Aplicamos a equalização apenas no canal Value mantendo
    # a cor e a saturação
    v_eq = clahe.apply(v)

    hsv_eq = cv2.merge((h, s, v_eq))
    img = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)

    return img


def process_input_images(path_file_value):
    processed = os.path.join(PROCESSED_DATASET_PATH, path_file_value)
    image_files = [
        f
        for f in os.listdir(os.path.join(DATASET_PATH, path_file_value))
        if f.endswith((".jpg"))
    ]

    os.makedirs(processed, exist_ok=True)
    for image_name in image_files:
        image = os.path.join(DATASET_PATH, path_file_value, image_name)
        image_object = cv2.imread(image)
        normalized = equalize_brightness_hsv(
            cv2.cvtColor(image_object, cv2.COLOR_BGR2RGB)
        )

        cv2.imwrite(os.path.join(processed, image), normalized)


def process_data():
    path_file = read_yaml(DATASET_YAML_PATH)

    process_input_images(path_file["train"])
    process_input_images(path_file["val"])

    logger.info("Imagens processadas com sucesso.")
    return True
