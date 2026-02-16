import logging
from pathlib import Path


def setup_llm_logging(log_file: str = "xxx.log", level: int = logging.DEBUG) -> None:
    """Set up unified logging for google-adk and LiteLLM to a file.

    Args:
        log_file: Path to the log file.
        level: Logging level (default: DEBUG to capture LLM calls).

    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Ensure the directory for the log file exists
    log_path = Path(log_file)
    if log_path.parent and not log_path.parent.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Loggers to configure
    loggers = [
        "google_adk.models.google_llm",  # ADK LLM calls
        "LiteLLM",  # LiteLLM calls
        "markov_agent",  # Markov Agent internal logs
    ]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        # Avoid duplicate handlers if setup is called multiple times
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the markov_agent namespace."""
    return logging.getLogger(f"markov_agent.{name}")
