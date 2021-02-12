import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from math import *
from pylab import *
import os
import pyfftw


def load_img(fpath, **kwargs):
    display = kwargs.pop('display', False)
    display_name = kwargs.pop('name', 'image')
    # read tiff image
    image = tiff.imread(fpath)
    # print('The shape of {} is {}'.format(display_name, image.shape))
    # real, imag, mag, pha = image[0, :, :], image[1, :, :], image[2, :, :], image[3, :, :]
    real, imag, mag, pha = image[0], image[1], image[2], image[3]
    cmplx_img = real + 1j * imag
    # display the complex image in a row
    if display:
        plt.subplot(141)
        plt.imshow(real, cmap='gray')
        plt.colorbar()
        plt.axis('off')
        plt.title('real part of {}'.format(display_name))
        plt.subplot(142)
        plt.imshow(imag, cmap='gray')
        plt.colorbar()
        plt.axis('off')
        plt.title('imaginary part of {}'.format(display_name))
        plt.subplot(143)
        plt.imshow(mag, cmap='gray')
        plt.axis('off')
        plt.colorbar()
        plt.title('magnitude of {}'.format(display_name))
        plt.subplot(144)
        plt.imshow(pha, cmap='gray')
        plt.colorbar()
        plt.axis('off')
        plt.title('phase of {}'.format(display_name))
        plt.show()
        plt.clf()

    return cmplx_img


def load_diffr_pattern(fpath, **kwargs):
    display = kwargs.pop("display", False)

    diffr_path = os.listdir(fpath)
    diffr_path.sort()
    diffr_ls = []
    for fname in diffr_path:
        dpattern = tiff.imread(os.path.join(fpath, fname))
        dpattern[dpattern < 0] = 0
        diffr_ls.append(dpattern)
    if display:
        plt.subplot(221)
        plt.imshow(diffr_ls[0], cmap='gray')
        plt.title('diffraction pattern')
        plt.axis('off')
        plt.colorbar()
        plt.subplot(222)
        dbscale = 20 * np.log10(diffr_ls[0]/np.amax(diffr_ls[0]))
        plt.imshow(dbscale, cmap='gray', vmax=0, vmin=-50)
        plt.title('diffraction pattern in decibel')
        plt.axis('off')
        plt.colorbar() 
        diffr_sqrt = np.sqrt(np.asarray(diffr_ls))
        plt.subplot(223)
        plt.imshow(diffr_sqrt[0], cmap='gray')
        plt.title('square root of diffraction pattern')
        plt.axis('off')
        plt.colorbar()
        plt.subplot(224)
        dbscale = 20 * np.log10(diffr_sqrt[0]/np.amax(diffr_sqrt[0]))
        plt.imshow(dbscale, cmap='gray', vmax=0, vmin=-50)
        plt.title('sqrt of diffr in decibel')
        plt.axis('off')
        plt.colorbar()
        plt.show()
        plt.clf()
    else:
        diffr_ls = np.sqrt(np.asarray(diffr_ls))

    return diffr_ls


def compute_translation(obj, num_agents, **kwargs):
    """
    Given the number of exposures and distance between the scan centers, returns
    the scan positions in raster order.
    :param obj: the simulation phantom.
    :param num_agents:  number of exposures, i.e. number of forward agents.
    :return: translations, i.e. absolute center positions of probes.
    """
    display = kwargs.pop("display", False)
    interval = kwargs.pop("interval", 32)
    m, n = np.shape(obj)
    x_num = int(np.sqrt(num_agents))
    y_num = int(np.sqrt(num_agents))
    position = [((i - x_num / 2 + 1 / 2) * interval + m / 2, (j - y_num / 2 + 1 / 2) * interval + n / 2)
                for j in range(x_num)
                for i in range(y_num)]
    if display:
        for j in range(num_agents):
            plt.plot(position[j][0], position[j][1], 'o')
        plt.title('translation')
        plt.xlim(0, obj.shape[0])
        plt.ylim(0, obj.shape[1])
        plt.show()
        plt.clf()

    return position


def compute_projection_mat(obj, probe, tran, **kwargs):
    display = kwargs.pop("display", False)
    # params
    x, y = obj.shape
    m, n = probe.shape
    num_agts = len(tran)
    # initialization
    projection_mat = np.zeros((num_agts, x, y))
    probe_mat = np.zeros((num_agts, x, y))
    for i in range(num_agts):
        projection_mat[i, int(tran[i][1] - m / 2):int(tran[i][1] + m / 2),
                          int(tran[i][0] - n / 2):int(tran[i][0] + n / 2)] = np.ones(probe.shape)
        probe_mat[i, int(tran[i][1] - m / 2):int(tran[i][1] + m / 2),
                     int(tran[i][0] - n / 2):int(tran[i][0] + n / 2)] = np.abs(probe)
    coverage = np.zeros(obj.shape)
    coverage[np.sum(projection_mat, 0) > 0] = 1
    if display:
        figure(num=None, figsize=(23.5, 5), dpi=100, facecolor='w', edgecolor='k')
        plt.subplot(141)
        for j in range(num_agts):
            plt.plot(tran[j][0], tran[j][1], 'o')
        plt.title('translation (d = 52)')
        plt.xlim(0, obj.shape[0])
        plt.ylim(0, obj.shape[1])
        plt.subplot(142)
        plt.imshow(np.sum(projection_mat, 0), cmap='gray')
        plt.colorbar()
        plt.axis('off')
        plt.title('sum of projections')
        plt.subplot(143)
        plt.imshow(np.sum(probe_mat, 0), cmap='gray')
        plt.colorbar()
        plt.axis('off')
        plt.title('sum of probe intensities')
        plt.subplot(144)
        plt.imshow(coverage, cmap='gray')
        plt.colorbar()
        plt.axis('off')
        plt.title('coverage')
        plt.show()
        plt.clf()

    return projection_mat, probe_mat, coverage


def build_init_guess(gt, probe, tran, diffr, **kwargs):
    method = kwargs.pop("method", 'formulated')
    display = kwargs.pop("display", False)
    patch_size = kwargs.pop("patch_sz", 256)
    obj_cstr = np.ones(gt.shape)
    probe_cstr = np.ones(probe.shape)
    projection_mat, _, _ = compute_projection_mat(gt, probe, tran, display=False)

    if method == 'zeros' or method == 'zero':
        obj_guess = np.zeros(gt.shape, dtype=complex) * obj_cstr
    elif method == 'ones' or method == 'one':
        obj_guess = np.ones(gt.shape, dtype=complex) * obj_cstr
    elif method == 'groundtruth' or method == 'ground truth':
        obj_guess = gt
    elif method == 'formulated':
        obj_guess = np.zeros(gt.shape, dtype=complex) * obj_cstr
        num_agents = len(diffr)
        m = patch_size
        n = patch_size
        Ny = float(np.sum(np.ones(diffr[1].shape)))
        Nd = float(np.sum(probe_cstr))
        Nx = float(patch_size * patch_size)
        Drms = np.sqrt(np.linalg.norm(probe) / Nd)
        patch_guess = np.zeros((num_agents, patch_size, patch_size), dtype=complex)
        for j in range(num_agents):
            Yrms = np.sqrt(np.linalg.norm(diffr[j]) / Ny)
            patch_guess[j] = np.sqrt(Ny / Nx) * Yrms / Drms
            obj_guess[int(tran[j][1] - m / 2):int(tran[j][1] + m / 2),
                      int(tran[j][0] - n / 2):int(tran[j][0] + n / 2)] += patch_guess[j]
        # (\sum P_j^t P_j)^(-1)
        norm = np.sum(np.abs(projection_mat) ** 2, 0)
        # (\sum P_j^t P_j)^(-1) \sum P_j^t X_j
        obj_guess = np.divide(obj_guess, norm, where=(norm != 0))
        obj_guess[obj_guess==0] = np.median(obj_guess)
    if display:
        plt.subplot(121)
        plt.imshow(np.abs(obj_guess), cmap='gray')
        plt.colorbar()
        plt.axis('off')
        plt.title('init guess amplitude')
        plt.subplot(122)
        plt.imshow(np.angle(obj_guess), cmap='gray')
        plt.colorbar()
        plt.axis('off')
        plt.title('init guess phase')
        plt.show()
        plt.clf()

    return obj_guess


def patch2img(img_patch, tran, norm, img_sz):
    """
    Project image patch back to the full size image and do the normalization.
    :param img_patch: image patches of size num_agts x m x n.
    :param tran: the absolute center of scan positions.
    :param norm: \Lambda ^{-1} which controls weights of pixels by the contribution to redundency.
    :param img_sz: size of image.
    :return: full-size complex image.
    """
    num_agts, m, n = img_patch.shape
    img = np.zeros(img_sz, dtype=complex)
    for j in range(num_agts):
        img[int(tran[j][1] - m / 2):int(tran[j][1] + m / 2),
            int(tran[j][0] - n / 2):int(tran[j][0] + n / 2)] += img_patch[j]
    output = np.divide(img, norm, where=norm != 0)

    return output


def img2patch(img, tran, patch_sz):
    """
    Convert full-size image to patches in line with scan positions and patch size.
    :param img: the full-size complex image.
    :param tran: the absolute center of scan positions.
    :param patch_sz: size of image patches, which is also the size of beam profile function.
    :return: image patches.
    """
    num_agts, m, n = patch_sz
    output = np.zeros(patch_sz, dtype=complex)
    for j in range(num_agts):
        output[j, :, :] = img[int(tran[j][1] - m / 2):int(tran[j][1] + m / 2),
                              int(tran[j][0] - n / 2):int(tran[j][0] + n / 2)]

    return output


def compute_ft(input):
    """ Calculate FFT of input """
    freq_domain = pyfftw.interfaces.numpy_fft.fft2(input, s=None, axes=(-2, -1), norm='ortho',
                                                   overwrite_input=False, planner_effort='FFTW_MEASURE', threads=1,
                                                   auto_align_input=True, auto_contiguous=True)
    freq_domain_shifted = np.fft.fftshift(freq_domain, axes=(1, 2))
    return freq_domain_shifted


def compute_ift(input):
    """ Calculate inverse FFT of input """
    freq_domain = np.fft.ifftshift(input, axes=(1, 2))
    spatial_domain = pyfftw.interfaces.numpy_fft.ifft2(freq_domain, s=None, axes=(-2, -1), norm='ortho',
                                                       overwrite_input=False, planner_effort='FFTW_MEASURE',
                                                       threads=1, auto_align_input=True, auto_contiguous=True)
    return spatial_domain
