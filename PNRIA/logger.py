import logging
import os
import sys
from typing import Optional

import torch
from PNRIA.utils.distributed import get_rank_num

def setup_logger(logger_name: str, gconfig, log_file="logger.log", debug=False, output_dir=None, run_name=None) -> logging.Logger:
    """
    Configure un logger pour un module spécifique avec des handlers pour la console et les fichiers.

    Args:
        logger_name (str): Nom du logger, souvent basé sur le nom de la classe ou du module.
        gconfig (dict): Configuration globale contenant 'output_dir' et 'run_name'.
        log_file (str): Nom du fichier de log.
        debug (bool): Mode debug activé si True.

    Returns:
        logging.Logger: L'instance configurée du logger.
    """
    logger = logging.getLogger(logger_name)

    config = gconfig
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False  # Empêche les logs de remonter à la racine

    # Format pour multi-GPU ou standard
    console_format = (
        f"[{logger_name} - GPU {get_rank_num()}] %(asctime)s - %(levelname)s - %(message)s"
        if torch.cuda.device_count() > 1
        else f"[{logger_name}] %(asctime)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_formatter = logging.Formatter(console_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    try:
        file_path = os.path.join(config["output_dir"] if output_dir else output_dir, config["run_name"] if run_name else run_name, log_file)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_handler = logging.FileHandler(file_path, mode="w+")
        file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        file_formatter = logging.Formatter(console_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning("Can't create log file: %s", e)

    # Désactiver les logs verbeux de `wandb` si disponible
    try:
        logging.getLogger("wandb").setLevel(logging.WARNING)
    except Exception:
        pass
    return logger
