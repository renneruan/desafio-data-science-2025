from pathlib import Path
import yaml

from smoke_detection import logger


def read_yaml(path_to_yaml: Path):
    """
    Lê um arquivo yaml e retorna um objeto com
    as informações lidas.

    Será utilizado para ler arquivos de configuração em formato yaml.

    Args:
        path_to_yaml (str): Caminho para o arquivo.

    Raises:
        ValueError: Se o arquivo estiver vazio
        Exception: Qualquer outra exceção

    Returns:
        Dict: Informações do arquivo em formato de dicionário
    """
    try:
        with open(path_to_yaml, encoding="UTF-8") as yaml_file:
            # Safe load é utilizado para evitar execução de código malicioso
            content = yaml.safe_load(yaml_file)
            logger.info(
                "Arquivo yaml: %s carregado com sucesso.", path_to_yaml
            )
            # print(f"Arquivo yaml: {path_to_yaml} carregado com sucesso.")
            return content
    except ValueError as exc:
        raise ValueError("Arquivo yaml está vazio.") from exc
    except Exception as e:
        raise e


def get_latest_folder(path: Path):
    """
    Retorna a última pasta criada em um diretório.

    Args:
        path (str): Caminho para o diretório.

    Returns:
        str: Nome da última pasta criada.
    """
    return max(path.iterdir(), key=lambda x: x.stat().st_ctime)
