from CTools.logger import save_context
from CTools.cli.flags import FLAGS
import numpy as np


def initContext(arguments):
    if isinstance(arguments, str):
        arguments = [arguments]
    KEY_ARGUMENTS = ["model_name"] + arguments
    text_logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, KEY_ARGUMENTS)

    seed_bias = FLAGS
    np.random.seed(1234 + seed_bias)

    try:
        initContext()
    except ImportError:
        print("No Torch Installed, Ignore Torch Init")
        pass

    return text_logger, MODELS_FOLDER, SUMMARIES_FOLDER


def initTorch(initSeed=1236):
    import torch

    seed_bias = FLAGS

    torch.manual_seed(initSeed + seed_bias)
    torch.cuda.manual_seed(initSeed + seed_bias + 1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FLAGS.device = device
