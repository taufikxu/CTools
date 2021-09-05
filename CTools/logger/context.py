import os
import pickle
import shutil

import time
import glob
import gpustat


from CTools.cli import flags
from CTools.cli.config import notValid
from CTools.logger.textLogger import build_logger

FLAGS = flags.FLAGS


def get_free_gpu():
    try:
        query = gpustat.new_query()
        num_gpus = len(query)
        vailable_gpus = []
        for i in range(num_gpus):
            if len(query[i].processes) == 0:
                vailable_gpus.append(str(i))
        vailable_gpus = vailable_gpus[: FLAGS.gpu_number]
        return ",".join(vailable_gpus)
    except OSError:
        return ""


def save_context(filename, keys):
    filename = os.path.splitext(filename)[0]
    # project = os.path.basename(os.getcwd())
    experiment_name = ""
    logfiles = []
    FILES_TO_BE_SAVED = ["./configs", "./library"]
    KEY_ARGUMENTS = keys

    if FLAGS.gpu.lower() not in ["-1", "none", notValid.lower()]:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    elif FLAGS.gpu_number != notValid and int(FLAGS.gpu_number) > 0:
        FLAGS.gpu_number = int(FLAGS.gpu_number)
        FLAGS.gpu = os.environ["CUDA_VISIBLE_DEVICES"] = get_free_gpu()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    configs_dict = FLAGS.get_dict()

    default_key = ""
    for item in KEY_ARGUMENTS:
        if isinstance(configs_dict[item], str) and "/" in configs_dict[item]:
            v = "path"
        else:
            v = configs_dict[item]
        default_key += "({}_{})".format(item.split(".")[-1], v)

    if FLAGS.results_folder == notValid:
        FLAGS.results_folder = "./0Results/"
    if FLAGS.subfolder != notValid:
        FLAGS.results_folder = os.path.join(FLAGS.results_folder, FLAGS.subfolder)

    experiment_name = "({data}_{file})_({time})_({default_key})_({user_key})".format(
        file=filename.replace("/", "_"),
        data=FLAGS.dataset,
        time=time.strftime("%Y-%m-%d-%H-%M-%S.%f"),
        default_key=default_key,
        user_key=FLAGS.key,
    )
    FLAGS.results_folder = os.path.join(FLAGS.results_folder, experiment_name)
    logfiles.append(os.path.join(FLAGS.results_folder, "Log.txt"))

    # if os.path.isabs(FLAGS.results_folder):
    #     experiment_name = "({})".format(project) + experiment_name
    #     FLAGS.results_folder = "({})".format(project) + FLAGS.results_folder
    #     logfiles.append("./Aresults/{}_log.txt".format(experiment_name))

    if os.path.exists(FLAGS.results_folder):
        raise FileExistsError("{} exits. Run it after a second.".format(FLAGS.results_folder))

    MODELS_FOLDER = FLAGS.results_folder + "/models/"
    SUMMARIES_FOLDER = FLAGS.results_folder + "/summary/"
    SOURCE_FOLDER = FLAGS.results_folder + "/source/"

    # creating result directories
    os.makedirs(FLAGS.results_folder)
    os.makedirs(MODELS_FOLDER)
    os.makedirs(SUMMARIES_FOLDER)
    os.makedirs(SOURCE_FOLDER)
    logger = build_logger(logfiles)

    destination = SOURCE_FOLDER
    posfix = ["py", "cpp", "h", "H"]

    for pos in posfix:
        for f in glob.glob("./*.{}".format(pos)):
            shutil.copy(f, os.path.join(destination, f))
    for f in FILES_TO_BE_SAVED:
        shutil.copytree(f, os.path.join(destination, f))

    configs_dict = FLAGS.get_dict()
    with open(os.path.join(SOURCE_FOLDER, "configs_dict.pkl"), "wb") as f:
        pickle.dump(configs_dict, f)
    return logger, MODELS_FOLDER, SUMMARIES_FOLDER
