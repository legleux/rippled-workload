# workload/logging_config.py

import logging
import logging.config
import os
import sys

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)-6s %(name)8s:%(lineno)d %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": sys.stdout,
        },
    },
    "loggers": {
        "workload": {
            "level": LOG_LEVEL,
            "handlers": ["console"],
            "propagate": False, # Don't pass 'workload' logs up to the root logger
        },
        # Shut the log levels for libraries up
        "fastapi": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        "uvicorn.access": {
             "level": "WARNING", # Quiets the noisy access logs
             "handlers": ["console"],
             "propagate": False,
        },
        "xrpl": {
            "level": "WARNING", # Only show warnings/errors from xrpl-py
            "handlers": ["console"],
            "propagate": False,
        }
    },
    # Default for all other loggers
    "root": {
        "level": "WARNING",
        "handlers": ["console"],
    },
}

def setup_logging():
    """ Apply the logging configuration. """
    logging.config.dictConfig(LOGGING_CONFIG)
    # Ensure output is unbuffered, essential for containers
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
