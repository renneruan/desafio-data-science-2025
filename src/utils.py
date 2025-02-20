from pathlib import Path
import yaml


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
            print(f"Arquivo yaml: {path_to_yaml} carregado com sucesso.")
            return content
    except ValueError as exc:
        raise ValueError("Arquivo yaml está vazio.") from exc
    except Exception as e:
        raise e
