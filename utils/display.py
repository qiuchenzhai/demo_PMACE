import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from math import *
from pylab import *
import os
from utils.utils import *
from utils.nrmse import *


def plot_nrmse(nrmse_ls, title, label, **kwargs):
    """

    :param nrmse_ls:
    :param title:
    :param label:
    :param kwargs:
    :return:
    """
    abscissa = kwargs.pop("abscissa", None)
    step_sz = kwargs.pop("step_sz", 1)
    display = kwargs.pop("display", False)
    save_fname = kwargs.pop("save_fname", None)
    fig_sz = kwargs.pop("fig_sz", [10, 4.8])
    xlabel, ylabel = label[0], label[1]
    figure(num=None, figsize=(fig_sz[0], fig_sz[1]), dpi=100, facecolor='w', edgecolor='k')
    if np.all(abscissa == None):
        if isinstance(nrmse_ls, dict):
            length = 0
            for line_label, line in nrmse_ls.items():
                length = np.maximum(length, len(line))
                plt.semilogy(line, label=line_label)
        else:
            line = np.asarray(nrmse_ls)
            length = len(line)
            line_label = label[2]
            plt.semilogy(line, label=line_label)
        plt.xticks(np.arange(length, step=step_sz), np.arange(1, length + 1, step=step_sz))
        plt.legend(loc='best')
        plt.legend(loc='best')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.grid(True)
        if save_fname is not None:
            plt.savefig('{}.png'.format(save_fname))
        if display:
            plt.show()
        plt.clf()
    else:
        # fig, ax = plt.subplots()
        # prop_cycle = plt.rcParams['axes.prop_cycle']
        # colors = prop_cycle.by_key()['color']
        if isinstance(nrmse_ls, dict):
            idx = 0
            # legend = []
            xmin = 1
            xmax = 0
            ymax = 0
            ymin = 1
            for line_label, line in nrmse_ls.items():
                # ax.plot(abscissa[idx], line, color=colors[idx])
                xmin = np.minimum(xmin, np.amin(np.asarray(abscissa[idx])))
                xmax = np.maximum(xmax, np.amax(np.asarray(abscissa[idx])))
                ymin = np.mininum(ymin, np.amin(np.asarray(line)))
                ymax = np.maximum(ymax, np.amax(np.asarray(line)))
                plt.semilogy(line, abscissa[idx], label=line_label)
                idx = idx + 1
                # legend.append(line_label)
        else:
            line = np.asarray(nrmse_ls)
            # ax.plot(abscissa, line)
            plt.semilogy(line, abscissa, label=label[2])
            xmin, xmax = np.amin(abscissa), np.max(abscissa)
            ymin, ymax = np.amin(line), np.amax(line)
            # legend = label[2]
        # ax.set_xlim([xmin, xmax])
        # ax.set_ymin([ymin, ymin])
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)
        # ax.legend(legend)
        plt.xlim([np.maximum(xmin, 0), xmax+1e-2])
        plt.ylim([ymin-1e-2, ymax+1e-2])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        if save_fname is not None:
            plt.savefig('{}.png'.format(save_fname))
        if display:
            plt.show()
        plt.clf()


def plot_img(img, ref_img, title, **kwargs):
    display = kwargs.pop("display", False)
    save_fname = kwargs.pop("save_fname", None)
    fig_sz = kwargs.pop("fig_sz", [6, 4.8])
    vmax = kwargs.pop("vmax", 1)
    vmin = kwargs.pop("vmin", 0)
    figure(num=None, figsize=(fig_sz[0], fig_sz[1]), dpi=200, facecolor='w', edgecolor='k')
    plt.subplot(221)
    plt.imshow(np.abs(img), cmap='gray', vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'{} obj density'.format(title))
    plt.subplot(222)
    plt.imshow(np.angle(img), cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.title(r'{} obj phase'.format(title))
    plt.subplot(223)
    plt.imshow(np.abs(img - ref_img), cmap='gray', vmin=np.amin(np.array([np.abs(img - ref_img)])),
               vmax=np.amax(np.array([np.abs(img - ref_img)])))
    plt.title(r'error amplitude')
    plt.colorbar()
    plt.axis('off')
    plt.subplot(224)
    ang_err = pha_err(ref_img, img)
    plt.imshow(ang_err, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.title(r'phase error')
    if save_fname is not None:
        plt.savefig('{}.png'.format(save_fname))
        cmplx_img = [np.real(img), np.imag(img), np.abs(img), np.angle(img)]
        tiff.imwrite('{}.tif'.format(save_fname), cmplx_img)
    if display:
        plt.show()
    plt.clf()

