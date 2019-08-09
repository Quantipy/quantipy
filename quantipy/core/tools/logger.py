
import logging

# formatter for logging.FileHandler()
FH_str = "%(asctime)s~%(name)s~%(funcName)s~%(levelname)s~%(message)s"
FH_formatter = logging.Formatter(FH_str)
# formatter for logging.StreamHandler()
SH_str = "%(levelname)s : %(name)s %(funcName)s --> %(message)s"
SH_formatter = logging.Formatter(SH_str)


def get_logger(name, path=None, level=logging.INFO, stream_handler=True):
    """
    Get a logger with agreed formats.

    Parameters
    ----------
    name: str
        Name of the logger, usually set to __name__
    path: str, default=None
        Location where the log-file is stored. If None the log will be
        generated in the same location as the originating script.
    level: logging level
        Set the logging level of this logger.
    stream_handler: bool, default True
        If true, a logging.StreamHandler is added to the logger
    sep: str
        A string that is added to the log file, when a logger is created.
    """
    if path is None:
        path = 'debug.log'
    if not level:
        level = logging.DEBUG
    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(path)
    logger.addHandler(file_handler)
    file_handler.setLevel(level)
    file_handler.setFormatter(FH_formatter)
    if stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(SH_formatter)
        logger.addHandler(stream_handler)
    return logger
