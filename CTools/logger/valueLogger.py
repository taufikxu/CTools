import os
import pickle
from collections import defaultdict

import matplotlib
import numpy as np

from matplotlib import pyplot as plt

from CTools.cli import flags
from torch.utils.tensorboard import SummaryWriter

FLAGS = flags.FLAGS
matplotlib.use("agg")


class Logger(object):
    def __init__(self, text_logger, log_dir="./logs", save_interval=100):
        self.stats = dict()
        self.text_logger = text_logger
        self.log_dir = log_dir
        self.save_interval = save_interval
        self.time2count = dict({})
        self.timecount = 0
        self.graph_count = defaultdict(int)
        self.tb_writter = SummaryWriter(log_dir)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def get_count(self, timestamp):
        if timestamp not in self.time2count:
            self.time2count[timestamp] = self.timecount
            self.timecount += 1
        return self.time2count[timestamp]

    def add(self, category, k, v, time, save=True):
        it = self.get_count(time)
        if category not in self.stats:
            self.stats[category] = {}
        if k not in self.stats[category]:
            self.stats[category][k] = []
        self.stats[category][k].append((it, v))
        self.tb_writter.add_scalar("{}/{}".format(category, k), v, it, walltime=time)
        if save is True and it % self.save_interval == 0:
            self.save()

    def addvs(self, category, keyvalue, time):
        it = self.get_count(time)
        for k in keyvalue:
            v = keyvalue[k]
            self.add(category, k, v, time, save=False)
        if it % self.save_interval == 0:
            self.save()

    def __getfilename(self, name, class_name=None, ext="png"):
        if class_name is None:
            class_name = "ImageVisualization"
        outdir = os.path.join(self.log_dir, class_name)

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if isinstance(name, str):
            outfile = os.path.join(outdir, "{}.{}".format(name, ext))
        else:
            outfile = os.path.join(outdir, "{}.{}".format(name, ext))
        return outfile

    def add_hist(self, vectors, name=None, **kwargs):
        if name is None:
            name = "Hist_{}".format(self.graph_count["hist"])
            self.graph_count["hist"] += 1
        outfile = self.__getfilename(name)
        _ = plt.figure()
        plt.hist(vectors)
        plt.savefig(outfile)
        plt.close()

    def add_contour(self, xrange, yrange, func, name=None, step=100, levels=10):
        if name is None:
            name = "contour_{}".format(self.graph_count["contour"])
            self.graph_count["contour"] += 1
        outfile = self.__getfilename(name)

        x = np.linspace(-xrange, xrange, step)
        y = np.linspace(-yrange, yrange, step)
        X, Y = np.meshgrid(x, y)
        inp = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], 1)
        value = func(inp).reshape(step, step)
        maxv, minv = np.max(value), np.min(value)
        levels = np.linspace(minv, maxv, levels)
        contour = plt.contour(X, Y, value, levels)
        plt.clabel(contour, fontsize=10)
        plt.savefig(outfile)
        plt.close()

    def add_heatmap(self, xrange, yrange, func, name=None, step=100):
        if name is None:
            name = "Heatmap_{}".format(self.graph_count["heatmap"])
            self.graph_count["heatmap"] += 1

        outfile = self.__getfilename(name)

        x = np.linspace(-xrange, xrange, step)
        y = np.linspace(-yrange, yrange, step)
        X, Y = np.meshgrid(x, y)
        inp = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], 1)
        value = func(inp).reshape(step, step)

        fig, ax = plt.subplots()
        z_min, z_max = np.min(value), np.max(value)
        c = ax.pcolormesh(X, Y, value, cmap="RdBu", vmin=z_min, vmax=z_max, shading="auto")
        ax.axis([X.min(), X.max(), Y.min(), Y.max()])
        fig.colorbar(c, ax=ax)
        plt.savefig(outfile)
        plt.close()

    def add_gif(self, img_list, name=None, **kwargs):
        if name is None:
            name = "Gif_{}".format(self.graph_count["gif"])
            self.graph_count["gif"] += 1

        outfile = self.__getfilename(name, ext="gif")
        img_list[0].save(outfile, save_all=True, append_images=img_list)

    def add_scatter(self, vectors, name=None, **kwargs):
        if name is None:
            name = "Scatter_{}".format(self.graph_count["scatter"])
            self.graph_count["scatter"] += 1

        outfile = self.__getfilename(name)
        _ = plt.figure()
        plt.scatter(vectors[:, 0], vectors[:, 1], **kwargs)
        plt.savefig(outfile)
        plt.close()

    def add_scatter_condition(self, vectors_dict, name=None, **kwargs):
        if name is None:
            name = "ScatterCond_{}".format(self.graph_count["scattercond"])
            self.graph_count["scattercond"] += 1

        outfile = self.__getfilename(name)
        _ = plt.figure()
        legend_names = []
        for k in vectors_dict:
            legend_names.append(k)
            vectors = vectors_dict[k]
            plt.scatter(vectors[:, 0], vectors[:, 1])
        plt.legend(legend_names)
        plt.savefig(outfile)
        plt.close()

    def log_info(self, prefix="", cats=None, log_level="info"):
        if cats is None:
            cats = self.stats.keys()

        if prefix != "":
            prefix += "\n"

        for cat in cats:
            prefix += "|{}: ".format(cat)
            for k in self.stats[cat]:
                prefix += "{}:{:.5f} ".format(k, self.stats[cat][k][-1][1])
            prefix += "\n"
        if log_level == "info":
            self.text_logger.info(prefix)
        else:
            raise ValueError("Not Suport other log-level")

    def save(self, filename="Stats.pkl"):
        filename = os.path.join(self.log_dir, filename)
        with open(filename, "wb") as f:
            pickle.dump(self.stats, f)
