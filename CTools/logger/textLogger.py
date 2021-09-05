import logging
import coloredlogs
import operator
import socket

from CTools.cli import flags

FLAGS = flags.FLAGS


def get_logger(logger_name=None):
    if logger_name is not None:
        logger = logging.getLogger(logger_name)
        logger.propagate = 0
    else:
        logger = logging.getLogger("taufikxu")
    return logger


def build_logger(file_names=None, logger_name=None):
    FORMAT = "%(asctime)s;%(levelname)s|%(message)s"
    DATEF = "%H-%M-%S"
    logging.basicConfig(format=FORMAT)
    logger = get_logger(logger_name)

    if isinstance(file_names, str):
        file_names = [file_names]

    if file_names is not None:
        for filename in file_names:
            fh = logging.FileHandler(filename=filename)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s;%(levelname)s|%(message)s", "%H:%M:%S")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    LEVEL_STYLES = dict(
        debug=dict(color="magenta"),
        info=dict(color="green"),
        verbose=dict(),
        warning=dict(color="blue"),
        error=dict(color="yellow"),
        critical=dict(color="red", bold=True),
    )
    coloredlogs.install(level=logging.INFO, fmt=FORMAT, datefmt=DATEF, level_styles=LEVEL_STYLES)

    def get_list_name(obj):
        if type(obj) is list:
            for i in range(len(obj)):
                if callable(obj[i]):
                    obj[i] = obj[i].__name__
        elif callable(obj):
            obj = obj.__name__
        return obj

    sorted_list = sorted(FLAGS.get_dict().items(), key=operator.itemgetter(0))
    host_info = "# " + ("%30s" % "Host Name") + ":\t" + socket.gethostname()
    logger.info("#" * 120)
    logger.info("----------Configurable Parameters In this Model----------")
    logger.info(host_info)
    logger.info("# " + ("%30s" % "GPU") + ":\t" + FLAGS.gpu)
    for name, val in sorted_list:
        logger.info("# " + ("%30s" % name) + ":\t" + str(get_list_name(val)))
    logger.info("#" * 120)
    return logger
