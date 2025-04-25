# -*- coding: utf-8 -*-
"""
Description: Utility functions for file system operations like loading text files,
             finding the project root, and loading environment variables from .env files.

Core Libraries/Concepts:
------------------------
- pathlib: For object-oriented filesystem paths.
  (https://docs.python.org/3/library/pathlib.html)
- python-dotenv: For loading .env files.
  (https://github.com/theskumar/python-dotenv)
- loguru: Logging library.

Key Functions:
--------------
- load_text_file: Reads content from a text file.
- get_project_root: Finds the project's root directory based on a marker file.
- load_env_file: Loads environment variables from a .env file in the project structure.

Sample I/O (load_text_file):
----------------------------
Input:
  file_path: "path/to/my_file.txt" (containing "Hello World")
Output:
  "Hello World"
"""

import os
from typing import Optional, Union # Import necessary types
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv


def load_text_file(file_path: Union[str, Path]) -> str:
    """
    Loads the content of a text file (e.g., AQL query) from the given path.

    Args:
        file_path (str): Relative or absolute path to the text file.

    Returns:
        str: Content of the text file as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there is an issue reading the file.
    """
    logger.debug(f"Attempting to load text file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            logger.success(
                f"Successfully loaded file: {file_path} (size: {len(content)} bytes)"
            )
            return content
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except IOError as e:
        logger.error(f"IOError while reading file {file_path}: {str(e)}")
        raise IOError(f"Error reading file {file_path}: {str(e)}") from e
    except Exception as e:
        logger.critical(f"Unexpected error loading file {file_path}: {str(e)}")
        raise


def get_project_root(marker_file: str = ".git") -> Path:
    """
    Find the project root directory by looking for a marker file.

    Args:
        marker_file (str): File/directory to look for (default: ".git")

    Returns:
        Path: Project root directory path

    Raises:
        RuntimeError: If marker file not found in parent directories
    """
    current_dir = Path(__file__).resolve().parent
    # Pylance incorrectly flags this comparison, the logic is sound.
    while current_dir != current_dir.root: # type: ignore
        if (current_dir / marker_file).exists():
            return current_dir
        current_dir = current_dir.parent
    raise RuntimeError(f"Could not find project root. Ensure {marker_file} exists.")


def load_env_file(env_type: Optional[str] = None) -> None:
    """
    Load environment variables from a .env file.

    Args:
        env_type (str, optional): Environment type suffix to look for.
            If provided, looks for .env.{env_type}. Otherwise looks for .env

    Raises:
        FileNotFoundError: If .env file not found in expected locations
    """
    project_dir = get_project_root()
    # Adjust search paths if necessary for this project structure
    env_dirs = [project_dir, project_dir.parent] # Example: search in utils dir parent and project root

    for env_dir in env_dirs:
        env_file = env_dir / (f".env.{env_type}" if env_type else ".env")
        if env_file.exists():
            logger.info(f"Loading environment variables from: {env_file}")
            load_dotenv(env_file, override=True)
            return

    logger.warning(
        f"Environment file {'.env.' + env_type if env_type else '.env'} "
        f"not found in expected locations: {env_dirs}. Proceeding without loading."
    )
    # Decide if not finding the file is an error or just a warning
    # raise FileNotFoundError(
    #     f"Environment file {'.env.' + env_type if env_type else '.env'} "
    #     f"not found in any known locations."
    # )