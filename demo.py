from utils.utils import *
from utils.pmace import *

'''
This file demonstrates the reconstruction of complex image by processing the simulated frame data using PMACE algorithm.
'''

# read ground truth images from file
path = 'data/'
object_path = path + 'SIM-generating_object.tiff'         # 4 x 660 x 660
obj = load_img(object_path, display=False, name='object')

probe_path = path + 'SIM-generating_probe.tiff'           # 4 x 256 x 256
probe = load_img(probe_path, display=False, name='probe')

x, y = obj.shape
m, n = probe.shape

# read frame data from file
diffr_path = 'data/frame_data/'
diffr = load_diffr_pattern(diffr_path, display=False)

# simulation parameters
num_agts = 64                                        
tran = compute_translation(obj, num_agts, interval=52, display=False)
projection_mat, probe_mat, coverage = compute_projection_mat(obj, probe, tran, display=False)

# reconstruction parameters
num_iter = 100
num_subiter = 3
alpha = 6
rho = 0.5

# # generate formulated initial guess for reconstruction
init_object = build_init_guess(obj, probe, tran, diffr, method='formulated', display=False)
# init_object = build_init_guess(obj, probe, tran, diffr, method='ground truth', display=False)

# perform PMACE reconstruction
print('========= PMACE reconstruction starts ==========')
obj_revy, _ = pmace_recon(init_object, diffr, tran, obj, probe,
                          num_iter=num_iter, num_subiter=num_subiter,
                          alpha=alpha, rho=rho, display=True)
