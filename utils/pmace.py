import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from math import *
import pyfftw
from copy import copy
from utils.utils import *
from utils.nrmse import *
from utils.display import *


def approx_map(obj_mat, dual_mat, diffr, init_cond, **kwargs):
    """
    Approximate proximal map operator with initial condition.
    For output-init-condition, we have w <- F(v; w) where the definition of approximal operator is given by:
    Def F(v; w)
        \hat{y} <- FZDv
        for k = 1,...K do {
            \theta <- FZDw / \left | FZDw  \right |
            w <- v + (\left | D \right | ^{2} + \alpha^{^{2}}I)^{\dagger} D^{*} Z^{t} F^{*} (y exp(i \theta) - \hat{y})
        }
        return w
    For input-init-condition, we have w <- F(v; v).
    :param obj_mat: image patches of size num_agents x m x n.
    :param dual_mat: beam profile function at each scan position.
    :param diffr: square root of measurements.
    :param init_cond: input-init-condition or output-init-condition.
    :param kwargs: parameters.
    :return: new estimation of image patches.
    """
    num_subiter = kwargs.pop("num_subiter", 2)
    param = kwargs.pop("param", 8)
    # FT{D*P_j*v}]
    fd = compute_ft(obj_mat * dual_mat)
    for k in range(num_subiter):
        # phase correction
        fd_init_cond = compute_ft(init_cond * dual_mat)
        # update image patch
        fd_update = diffr * np.exp(1j * np.angle(fd_init_cond)) - fd
        img_patch = compute_ift(fd_update)
        init_cond = obj_mat + np.reciprocal(np.abs(dual_mat) ** 2 + param ** 2) * np.conj(dual_mat) * img_patch

    return init_cond


def consen_operator(obj_mat, tran, norm, img_sz):
    """
    The consensus operator G \left ( w \right ) = \begin{bmatrix}
                                                      \bar{w_{0}}\\
                                                      \vdots \\
                                                      \bar{w_{J-1}}
                                                   \end{bmatrix}
    where \bar{w_{j}} = P_{j} \Lambda ^{-1} \sum_{j=1}^{J-1} P_{j}^{t} w_{j}.
    :param obj_mat: image patches of size num_agents x m x n.
    :param tran: the absolute center position of each scan.
    :param norm: \Lambda ^{-1} which controls weights of pixels by the contribution to redundancy.
    :param img_sz: size of full image.
    :return: new estimation of image patches.
    """
    # obtain complex image by processing the patches
    cmplx_img = patch2img(obj_mat, tran, norm, img_sz)
    # extract patches out of image
    output_mat = img2patch(cmplx_img, tran, obj_mat.shape)

    return output_mat


def pmace_recon(init_guess, diffr, tran, obj_ref, probe, **kwargs):
    """
    Given the known beam profile function, reconstruct complex image from data files using
    PMACE algorithm with output-init-condition:
        Initialization
        While not converged do {
            w <- F(v; w)
            z <- G(2w - v)
            v <- v + 2 \rho (z - w)
        }
        return x <- v
    :param init_guess: initial guess of complex image.
    :param diffr: square root of recorded diffraction patterns.
    :param tran: the absolute center of each scan position.
    :param obj_ref: the ground truth image.
    :param probe: the beam profile function.
    :param kwargs: other arguments.
    :return: image estimation and error sequence.
    """
    num_iter = kwargs.pop("num_iter", 100)
    num_subiter = kwargs.pop("num_subiter", 2)
    alpha = kwargs.pop("alpha", 8)
    rho = kwargs.pop("rho", 0.9)
    display = kwargs.pop("display", False)
    # iniitialization
    num_agts = len(diffr)
    # x, y = init_guess.shape
    m, n = probe.shape
    proj_mat, _, coverage = compute_projection_mat(obj_ref, probe, tran, display=False)
    consen_norm = np.sum(np.abs(proj_mat) ** 2, 0)
    x_mat = img2patch(init_guess, tran, [num_agts, m, n])
    u_mat = [probe] * num_agts
    v_mat = np.copy(x_mat)
    obj_nrmse = []
    # time_ls = []
    # start_time = time.time()
    # reconstruction with known probe
    for i in range(num_iter):
        # w <- F(v; w)
        x_mat = approx_map(v_mat, u_mat, diffr, init_cond=x_mat, num_subiter=num_subiter, param=alpha)

        # z <- G(2w - v)
        x_cons = consen_operator(2 * x_mat - v_mat, tran, consen_norm, init_guess.shape)

        # v <- v + 2 \rho (z - w)
        v_mat += 2 * rho * (x_cons - x_mat)

        # calculte result at the end of current iteration
        obj_revy = patch2img(v_mat, tran, consen_norm, init_guess.shape)
        # clear the edges
        obj_ref[coverage == 0] = 0 + 1j * 0
        obj_revy[coverage == 0] = 0 + 1j * 0
        obj_revy = phase_norm(obj_revy, obj_ref)
        # # calculate time consumption
        # elapsed_time = time.time() - start_time
        # time_ls.append(elapsed_time)
        # calculate nrmse
        nrmse_val = calculate_nrmse(obj_revy, obj_ref, coverage)
        obj_nrmse.append(nrmse_val)
        print('nrmse val = {} at {}th iteration'.format(nrmse_val, i+1))

    # plot or save reconstructed complex image
    path = 'results/'
    plot_img(obj_revy, obj_ref,
             title='PMACE', vmax=0.85, vmin=0,
             display=display, save_fname=path+'PMACE_K_{}_alpha_{}'.format(num_subiter, alpha))
    # plot or save convergence plot
    xlabel, ylabel = 'number of Mann iteration', 'NRMSE value'
    line_label = r'pmace_nrmse(K={}, $\alpha$={}, $\rho$={})'.format(num_subiter, alpha, rho)
    plot_nrmse(obj_nrmse,
               title='NRMSE plot of PMACE algorithm', label=[xlabel, ylabel, line_label],
               step_sz=15, fig_sz=[8, 4.8],
               display=display, save_fname=path+'PMACE_K_{}_alpha_{}_nrmse'.format(num_subiter, alpha))

    return obj_revy, obj_nrmse


