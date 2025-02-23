"""
Módulo para armazenamento de constantes dos caminhos de configuração

Armazena valores de:

- MODEL_PATH: Caminho para arquivo do modelo treinado.
"""

from pathlib import Path

MODEL_PATH = Path("artifacts/model/best.pt")
DATASET_YAML_PATH = Path("datasets/data/data.yaml")
OUTPUT_FOLDER = Path("runs/detect")
