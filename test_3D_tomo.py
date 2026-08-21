# import os, getpass, sys
# import argparse
import numpy as np
import h5py
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import scipy
# from scipy.sparse import diags
from scipy.stats import multivariate_normal
from scipy.interpolate import RegularGridInterpolator
from kick_matrix_regularisation import read_kick_matrix, make_kick_regularisation_matrix, plot_kicks, plot_precision_matrix
from scipy.constants import physical_constants as const
import matplotlib.pyplot as plt
import cuqi
from cuqi.implicitprior import NonnegativeGaussian, RegularizedGaussian
from cuqi.model import LinearModel
from cuqi.sampler._conjugate import Conjugate
from cuqi.sampler._rto import LinearRTO, RegularizedLinearRTO
from cuqi.sampler._gibbs import HybridGibbs
from cuqi.utilities import count_nonzero
# from cuqi.problem import BayesianProblem
from scipy.io import netcdf

from functions_for_3D_tomo import CustomConjugatePair

def P_canonical(Energy, pitch,
                R, Z,
                mass, q,
                psi, Bphi, B):
    
    # Energy is given in keV
    v = np.sqrt(2*Energy*1e3*q / mass)
    vpar = v * pitch 

    return mass*R*Bphi*vpar / B + q*psi

def mu(Pphi,
       Energy,
       R, Z,
       mass, q,
       psi, Bphi, B):
    
    # Energy is given in keV
    return Energy*1e3*q/B - B/(2*mass) * ( (Pphi - q*psi) / (R*Bphi) )**2

def PSNR(original, compressed):
    normalisation = np.max([np.max(original), np.max(compressed)])
    mse = np.mean((original/normalisation - compressed/normalisation) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0 # For 8-bit number
    max_pixel = 1.7976931348623158e+308 # For 64-bit 
    max_pixel = 1 # For double precision vector elements
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def make_test_distribution(Eaxis, Laxis, Paxis, mean=[10.0, 0.5, 0.5], std=[1.0, 0.1, 0.1], rot=0):

    # Make a Gaussian blob centered around 'center'
    E, L, P = np.meshgrid(Eaxis, Laxis, Paxis, indexing='ij')
    ELP = np.column_stack([E.flat, L.flat, P.flat])
    
    # 1. axis is Energy, 2. axis is Lambda, 3. axis is Pphi
    # Therefore, a positive angle is ctr-clockwise from the vertical Lambda axis
    rot_phi = np.pi/3
    rot_mat = np.array([[1, 0, 0],[0, np.cos(rot_phi), -np.sin(rot_phi)],[0, np.sin(rot_phi), np.cos(rot_phi)]])
    if rot == 0:
        covariance = np.diag(np.array(std)**2)
    elif rot == 1:
        std = [1, 0.05, 0.15]
        covariance = rot_mat.T @ np.diag(np.array(std)**2) @ rot_mat

    z = multivariate_normal.pdf(ELP, mean=mean, cov=covariance)

    # Reshape back to a (30, 30) grid.
    dist = z.reshape(E.shape)
    
    return dist

def make_L1_regularisation_matrix(Eaxis, muaxis, Pphiaxis, ENBI):
    
    # Grid spacing
    deltaE = np.mean(np.diff(Eaxis[len(Eaxis)//2:] / ENBI)) # Normalise with NBI energy such that all dimensions are of order 1.
    deltaPphi = np.mean(np.diff(Pphiaxis))
    deltamu = np.mean(np.diff(muaxis))
    # deltaE    = 1.0
    # deltaPphi = 1.0
    # deltamu   = 1.0
    
    LE = scipy.sparse.csr_matrix(scipy.sparse.diags([-1,1],[0,1], shape=(len(Eaxis),len(Eaxis)))) / deltaE
    LP = scipy.sparse.csr_matrix(scipy.sparse.diags([-1,1],[0,1], shape=(len(Pphiaxis),len(Pphiaxis)))) / deltaPphi
    Lm = scipy.sparse.csr_matrix(scipy.sparse.diags([-1,1],[0,1], shape=(len(muaxis),len(muaxis)))) / deltamu
    # LE = scipy.sparse.diags([-1,1],[0,1], shape=(len(Eaxis),len(Eaxis))).todense()
    # LP = scipy.sparse.diags([-1,1],[0,1], shape=(len(Pphiaxis),len(Pphiaxis))).todense()
    # Lm = scipy.sparse.diags([-1,1],[0,1], shape=(len(muaxis),len(muaxis))).todense()
    # # If sparse matrix
    # LE = scipy.sparse.csr_matrix(scipy.sparse.diags([-1,1],[0,1], shape=(len(Eaxis),len(Eaxis))))
    # LP = scipy.sparse.csr_matrix(scipy.sparse.diags([-1,1],[0,1], shape=(len(Pphiaxis),len(Pphiaxis))))
    # Lm = scipy.sparse.csr_matrix(scipy.sparse.diags([-1,1],[0,1], shape=(len(muaxis),len(muaxis))))
    #L1 = np.concatenate((np.kron(LE,np.kron(np.eye(len(muaxis)),np.eye(len(Pphiaxis)))),
    #                     np.kron(np.eye(len(Eaxis)),np.kron(Lm,np.eye(len(Pphiaxis)))),
    #                     np.kron(np.eye(len(Eaxis)),np.kron(np.eye(len(muaxis)),LP))), axis=0)
    L1 = scipy.sparse.csr_matrix(scipy.sparse.vstack([scipy.sparse.kron(scipy.sparse.eye(len(Eaxis)), scipy.sparse.kron(scipy.sparse.eye(len(muaxis)), LP)), 
                                scipy.sparse.kron(scipy.sparse.eye(len(Eaxis)), scipy.sparse.kron(Lm, scipy.sparse.eye(len(Pphiaxis)))), 
                                scipy.sparse.kron(LE, scipy.sparse.kron(scipy.sparse.eye(len(muaxis)), scipy.sparse.eye(len(Pphiaxis))))]))
    
    return L1


def interpolate_kick_matrix(kick_dict, Eaxis, Laxis, Paxis, ENBI):
    kick_prob_array = kick_dict['kick_prob_array']
    dEaxis = kick_dict['dEaxis']
    dPphiaxis = kick_dict['dPphiaxis'] # I think the Kick model Pphi axis goes from -1 to 0, and not 0 to 1
    Eax = kick_dict['Eaxis']
    Lax = kick_dict['muaxis']
    Pax = kick_dict['Pphiaxis'] # I think the Kick model Pphi axis goes from -1 to 0, and not 0 to 1
    dEarray = np.array(dEaxis) / ENBI
    dParray = np.array(dPphiaxis)
    kick_phase_space = kick_dict['kick_phase_space']
    dE_kick_phase_space = kick_dict['dE_kick_phase_space']
    dPphi_kick_phase_space = kick_dict['dPphi_kick_phase_space']
    interp_kick_strength = RegularGridInterpolator((Eax, Lax, -np.flip(Pax)), np.flip(kick_phase_space, axis=2), method='linear', bounds_error=False, fill_value=0)
    interp_dE_kick = RegularGridInterpolator((Eax, Lax, -np.flip(Pax)), np.flip(dE_kick_phase_space, axis=2), method='linear', bounds_error=False, fill_value=0)
    interp_dPphi_kick = RegularGridInterpolator((Eax, Lax, -np.flip(Pax)), -np.flip(dPphi_kick_phase_space, axis=2), method='linear', bounds_error=False, fill_value=0)
    EAXIS, LAXIS, PAXIS = np.meshgrid(Eaxis, Laxis, Paxis, indexing='ij')
    kick_strength = interp_kick_strength((EAXIS, LAXIS, PAXIS))
    dE_kick = interp_dE_kick((EAXIS, LAXIS, PAXIS))
    dPphi_kick = interp_dPphi_kick((EAXIS, LAXIS, PAXIS))
    del EAXIS, LAXIS, PAXIS

    # Convert kick_prob_array
    kick_new_array = np.zeros(kick_prob_array.shape)
    kick_new_array[:, 0] = kick_prob_array[:, 0]
    kick_new_array[:, 1] = kick_prob_array[:, 1]
    for idx in range(kick_prob_array.shape[0]): # !!! The indices are not exactly being interpolated, just placed in a new grid !!!
        Eidx  = int(kick_prob_array[idx, 2]) - 1
        Pidx  = int(kick_prob_array[idx, 3]) - 1
        midx  = int(kick_prob_array[idx, 4]) - 1
        kick_new_array[idx, 2] = np.argmin(np.abs(Eaxis - Eax[Eidx])) + 1 # To match the 'make_kick_regularisation_matrix' function
        kick_new_array[idx, 3] = np.argmin(np.abs(Paxis - (-Pax[Pidx]))) + 1 # To match the 'make_kick_regularisation_matrix' function
        kick_new_array[idx, 4] = np.argmin(np.abs(Laxis - Lax[midx])) + 1 # To match the 'make_kick_regularisation_matrix' function
        kick_new_array[idx,5] = np.sqrt(dEarray[int(kick_prob_array[idx,0])-1]**2 + dParray[int(kick_prob_array[idx,1])-1]**2)

    kick_dict_new = {'kick_prob_array': kick_new_array, 'kick_phase_space': kick_strength, 
                        'dE_kick_phase_space': dE_kick, 'dPphi_kick_phase_space': dPphi_kick, 
                        'dEaxis': dEaxis, 'dPphiaxis': -np.flip(dPphiaxis),
                        'Eaxis': Eaxis, 'Pphiaxis': Paxis, 'muaxis': Laxis}

    return kick_dict_new

def plot_3D_inversions(inversions, true_dist, Eaxis, Laxis, Paxis, weights=None, unc=None, prec=None):

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # Unpack various variables dict
    various_variables_dict = h5py.File('C:/Users/larsen/Documents/Analytical orbit tomography/Resonant mode-particle interactions/Tomography/FIDA/86327/Time_0-48/various_variables_dict_86327_0.480.h5', 'r')
    #Pphi_axis = various_variables_dict['Pphi_axis'][()]
    Pphi_trap = various_variables_dict['Pphi_trap'][()]
    #q = various_variables_dict['q'][()]
    #m = various_variables_dict['m'][()]
    #psi = various_variables_dict['psi'][()]
    psiwall = various_variables_dict['psiwall'][()]
    #psiwall_norm = various_variables_dict['psiwall_norm'][()]
    #psi0 = various_variables_dict['psi0'][()]
    #B = various_variables_dict['Babs'][()]
    B0 = various_variables_dict['B0'][()]
    B_LFS = various_variables_dict['B_LFS'][()]
    B_HFS = various_variables_dict['B_HFS'][()]
    #Bphi = various_variables_dict['Bphi'][()]
    Bphi0 = various_variables_dict['Bphi0'][()]
    Bphi_LFS = various_variables_dict['Bphi_LFS'][()]
    Bphi_HFS = various_variables_dict['Bphi_HFS'][()]
    B_trap = various_variables_dict['B_trap'][()]
    #R = various_variables_dict['R'][()]
    R_LFS = various_variables_dict['R_LFS'][()]
    R_HFS = various_variables_dict['R_HFS'][()]
    R0 = various_variables_dict['R0'][()]
    Z_LFS = various_variables_dict['Z_LFS'][()]
    Z_HFS = various_variables_dict['Z_HFS'][()]
    #Rvec = various_variables_dict['Rvec'][()]
    #zvec = various_variables_dict['zvec'][()]
    #axisR = various_variables_dict['axisR'][()]
    #axisZ = various_variables_dict['axisZ'][()]
    pitch = various_variables_dict['pitch'][()]
    various_variables_dict.close()
    curdir = -1
    
    if prec is None:
        min_color = np.min(inversions)
        min_color = 0
        max_color = np.max([np.max(inversions), np.max(true_dist)])
        max_color = np.max(true_dist)
    else:
        min_color = -np.max(np.abs(inversions))
        max_color = np.max(np.abs(inversions))
        cmap = 'seismic'
    if (unc is None) & (prec is None):
        cmap = 'hot'
    elif (unc is not None) & (prec is None):
        cmap = 'turbo'
        min_color = np.min(inversions)
        max_color = np.max(np.abs(inversions))
    if weights == True:
        min_color = 0.0
        max_color = None
        
    for i in range(inversions.shape[1]+1):

        # Reshape inversion
        if i < inversions.shape[1]:
            recon = inversions[:,i].reshape((len(Eaxis), len(Laxis), len(Paxis)))

        fig = plt.figure()
        ax  = fig.add_subplot(111)    # The big subplot
        ax1 = fig.add_subplot(4, 4, 1)
        ax2 = fig.add_subplot(4, 4, 2)
        ax3 = fig.add_subplot(4, 4, 3)
        ax4 = fig.add_subplot(4, 4, 4)
        ax5 = fig.add_subplot(4, 4, 5)
        ax6 = fig.add_subplot(4, 4, 6)
        ax7 = fig.add_subplot(4, 4, 7)
        ax8 = fig.add_subplot(4, 4, 8)
        ax9 = fig.add_subplot(4, 4, 9)
        ax10 = fig.add_subplot(4, 4, 10)
        ax11 = fig.add_subplot(4, 4, 11)
        ax12 = fig.add_subplot(4, 4, 12)
        ax13 = fig.add_subplot(4, 4, 13)
        ax14 = fig.add_subplot(4, 4, 14)
        ax15 = fig.add_subplot(4, 4, 15)
        ax16 = fig.add_subplot(4, 4, 16)
        axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16]
        
        # E_subset = Eaxis[(len(Eaxis)//2):] # Positive energies
        E_subset = Eaxis[:(len(Eaxis)//2)] # Negative energies
        
        counter = -1
        for eidx, E in enumerate(E_subset):
            e = np.argmin(np.abs(Eaxis - E))
            if true_dist.shape[0] == Eaxis.size//2:
                # Then the true distribution only includes sigma = +1 or sigma = -1, already
                e_dist = np.argmin(np.abs(E_subset - E))
            elif true_dist.shape[0] == Eaxis.size:
                # Then true_dist inlcudes all energies
                e_dist = np.argmin(np.abs(Eaxis - E))
            
            counter += 1
            if E < 0.0:
                ax_idx = len(axs) - counter - 1
            else:
                ax_idx = counter
            
            Lambda_trap = (np.abs(E)*1e3*Qe/B_trap)*B0/np.abs(E*Qe*1e3)
            
            Pphi_LFS = P_canonical(np.abs(E), pitch, R_LFS, Z_LFS, mD, Qe, psiwall, Bphi_LFS, B_LFS)
            Lambda_LFS = mu(Pphi_LFS, np.abs(E), R_LFS, Z_LFS, mD, Qe, psiwall, Bphi_LFS, B_LFS) * B0/np.abs(E*Qe*1e3)
            Pphi_HFS = P_canonical(np.abs(E), pitch, R_HFS, Z_HFS, mD, Qe, psiwall, Bphi_HFS, B_HFS)
            Lambda_HFS = mu(Pphi_HFS, np.abs(E), R_HFS, Z_HFS, mD, Qe, psiwall, Bphi_HFS, B_HFS) * B0/np.abs(E*Qe*1e3)
            Pphi0 = P_canonical(np.abs(E), pitch, R0, 0, mD, Qe, 0, Bphi0, B0)
            Lambda0 = mu(Pphi0, np.abs(E), R0, 0, mD, Qe, 0, Bphi0, B0) * B0/np.abs(E*Qe*1e3)
            # Plot parabolas and trapped boundary
            axs[ax_idx].plot(Pphi_LFS/Qe/(psiwall*curdir),Lambda_LFS,'w-')
            axs[ax_idx].plot(Pphi_HFS/Qe/(psiwall*curdir),Lambda_HFS,'w--')
            axs[ax_idx].plot(Pphi_trap/Qe/(psiwall*curdir),Lambda_trap,'w-')
            axs[ax_idx].plot(Pphi0/Qe/(psiwall*curdir),Lambda0,'w-.')
            axs[ax_idx].plot(-np.ones(10),np.linspace(np.min(Lambda_trap),np.max(Lambda_trap),10),'w-')
            
            if i < inversions.shape[1]:
                im = axs[ax_idx].imshow(np.fliplr(recon[e, :, :]), 
                    extent=(-Paxis[-1], -Paxis[0], Laxis[0], Laxis[-1]), vmin=min_color, vmax=max_color, cmap=cmap, origin='lower')
            else:
                im = axs[ax_idx].imshow(np.fliplr(true_dist[e_dist, :, :]), 
                    extent=(-Paxis[-1], -Paxis[0], Laxis[0], Laxis[-1]), vmin=min_color, vmax=max_color, cmap=cmap, origin='lower')
            axs[ax_idx].set_xlim([-Paxis[-1], -Paxis[0]])
            divider = make_axes_locatable(axs[ax_idx])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            # fig.colorbar(im, cax=cax, orientation='vertical', extend='max')
            axs[ax_idx].set_title(f'Energy, E={np.abs(np.round(E,1))} keV')
            if ax_idx < (len(Eaxis[(len(Eaxis)//2):])-3):
                axs[ax_idx].set_xticks([])

        # Turn off axis lines and ticks of the big subplot
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        ax.set_ylabel('Norm. magnetic moment, $\\Lambda=\\mu B_{0}/E$', fontsize=14)
        ax.set_xlabel('Norm. tor. can. angular momentum, $P_{\\phi}/q|\\Psi_{w}|$', fontsize=14)
        ax.set_title('Systems 5, 15 and 19, all channels', fontsize=14)
        plt.ion()

    plt.show()

def tikhonov_3D_reg(s, W, alpha, L=None):

    if alpha is None:
        alpha = np.logspace(-2, 2, 5)

    inversions = np.zeros((W.shape[1],len(alpha)))

    Wnorm = W / np.linalg.norm(W)
    snorm = s / np.linalg.norm(s)
    normalisation = np.linalg.norm(W) / np.linalg.norm(s)

    if L is None:
        for i, a in enumerate(alpha):
            print(f'Computing inversions using zeroth-order Tikhonov regularisation, alpha={a}')
            clf = Ridge(alpha=a, positive=True)
            clf.fit(Wnorm, snorm)

            inversions[:,i] = clf.coef_ / normalisation
    elif isinstance(L, int):
        print('\nBeware this takes a LONG time and uses A LOT of CPU resources, and quite a lot of memory... (even with sparse matrices)')
        Lp = scipy.sparse.diags([-1,1],[-1,0], shape=(35,35))
        Ll = scipy.sparse.diags([-1,1],[-1,0], shape=(30,30))
        LE = scipy.sparse.diags([-1,1],[0,1], shape=(24,24))
        L = scipy.sparse.vstack([scipy.sparse.kron(scipy.sparse.eye(len(Eaxis)), scipy.sparse.kron(scipy.sparse.eye(len(Laxis)),Lp)), 
                                scipy.sparse.kron(scipy.sparse.eye(len(Eaxis)), scipy.sparse.kron(Ll,scipy.sparse.eye(len(Paxis)))),
                                scipy.sparse.kron(LE, scipy.sparse.kron(scipy.sparse.eye(len(Laxis)),scipy.sparse.eye(len(Paxis))))])
        del Lp, Ll, LE

        for i, a in enumerate(alpha):
            print(f'Computing inversions using first-order Tikhonov regularisation, alpha={a}')
            WW = scipy.sparse.vstack([scipy.sparse.bsr_matrix(Wnorm), a * L])
            ss = np.vstack([snorm[:,None], np.zeros((L.shape[0]))[:,None]]).flatten()
            inversions[:,i] = scipy.optimize.lsq_linear(WW, ss, bounds=(0, np.inf), verbose=2).x / normalisation
    else:
        for i, a in enumerate(alpha):
            print(f'Computing inversions using kick-model regularisation, alpha={a}')
            print('\nEven with sparse W and L matrices, this takes up A LOT of CPU resources!')
            WW = scipy.sparse.vstack([Wnorm, a * L])
            ss = np.vstack((snorm[:,None], np.zeros((L.shape[0]))[:,None])).flatten()
            inversions[:,i] = scipy.optimize.lsq_linear(WW, ss, bounds=(0, np.inf), verbose=2).x / normalisation

    return inversions

def read_TRANSP_cdf(foldername='C:/Users/larsen/Documents/Analytical orbit tomography/Resonant mode-particle interactions/TRANSP fast-ion distributions/With_NBI2_ctr-going/' ,filename='86327P55_fi_5.cdf'):
    
    cdf_file_read = netcdf.netcdf_file(foldername+filename, 'r')
    FIdist_temp = cdf_file_read.variables['F_D_NBI'][:].T
    FIdist = FIdist_temp.copy()*1e3 # Multiply with 1e3 to get units per keV instead of eV. The units are #/cm^3/ev/d(Omega/4pi)
    R2D_temp = cdf_file_read.variables['R2D'][:]
    R2D = R2D_temp.copy()
    Z2D_temp = cdf_file_read.variables['Z2D'][:]
    Z2D = Z2D_temp.copy()
    pTRANSP_temp = cdf_file_read.variables['A_D_NBI'][:]
    pTRANSP = pTRANSP_temp.copy()
    ETRANSP_temp = cdf_file_read.variables['E_D_NBI'][:]/1e3 # Divide by 1e3 to get in keV
    ETRANSP = ETRANSP_temp.copy()
    ETRANSP = ETRANSP.reshape((len(ETRANSP)))
    pTRANSP = pTRANSP.reshape((len(pTRANSP)))
    time = cdf_file_read.variables['TIME'][()].copy()
    
    return FIdist, R2D, Z2D, ETRANSP, pTRANSP, time

# Make hyperprior for Gazzola rotate-then-scale method
def precision_matrix(spara, L):
    
    prec = ((L.T) @ scipy.sparse.kron(np.diag([1, 1, spara]), scipy.sparse.eye((L.shape[0]//3))) @ L).todense()
    
    return prec

# =================================================================================================
# =================================================================================================

'''

Testing 3D inversions on TCV.

'''
# Optional arguments
experimental = 0 # Use experimental data or not
ASCOT = 0 # Use ASCOT distribution and FIDASIM spectra
TRANSP = 0 # Use TRANSP disitribution
kicktime = 6 # Whether to use TRANSP distribution before or after kick (1 or 2) or (5 or 6). At t=0.48s kicktime=5 (before kick). Number 6 is then approximately at t=51 (after kick).
co_going = 'projection' # True for co, False for ctr, 'projection' if projecting true dist onto right singular vectors
kick_reg = 0 # Whether to use kick matrix as regularisation
Gazzola = 1 # Use Gazzola method of rotation and scaling instead of projection
L1 = 0 # Whether to use first-order Tikhonov regularisation
alpha = np.logspace(-10, -6, 5) # Regularisation parameters
cuqipy = 1 # Whether to use CUQIpy or not
hmc = 0 # Whether to use hamiltorch for Hamiltonian Monte Carlo sampling of posterior
fild = 0 # Whether to invert FILD data or not
include_CII_spectra = False # True if we use measurements at wavelengths that are otherwise corrupted by the CII lines in experiments
hyperparam = False # If including strength of smooting as hyperparameter in Gazzola method
nburn = 100
nsamples = 500
sys_list = [5, 15]
ch_list = np.linspace(1,20,20)
ENBI = 24.0

mD = const['deuteron mass'][0]     # Deuterium [kg]
me = const['electron mass'][0]
Qe = const['elementary charge'][0]

print('---------- Load various variables to get psiwall ----------')
various_variables_dict = h5py.File('C:/Users/larsen/Documents/Analytical orbit tomography/Resonant mode-particle interactions/Tomography/FIDA/86327/Time_0-48/various_variables_dict_86327_0.480.h5', 'r')
psiwall = various_variables_dict['psiwall'][()]
various_variables_dict.close()
print('---------- Load weight functions ----------')
h = h5py.File("C:/Users/larsen/Documents/Analytical orbit tomography/Resonant mode-particle interactions/Phase-space weights/86327/time_0-48/CoMweights_highres.h5", 'r')
W = h['weights'][()]
Eaxis = h['energy'][()] / Qe / 1e3
Laxis = h['normmagmom'][()]
Paxis = h['pphi'][()] / Qe / psiwall
orbits = h['orbits'][()]

# # Plot some weight functions
# from functions_for_3D_tomo import plot_CoM_weight_functions
# wavelengths = np.array([653, 654, 655, 657, 658, 659])
# plot_CoM_weight_functions(h, wavelengths, ENBI)

h.close()
# Make mask
for i in range(orbits.size):
    page, row, col = np.unravel_index(i, shape=orbits.shape)
    if len(orbits[page,row,col]) > 0:
        orbits[page,row,col] = 1.0
        try:
            indices.append(i) # Linear indices in full phase-space corresponding to a valid and confined orbit
        except NameError:
            indices = [i]
    elif len(orbits[page,row,col]) == 0:
        orbits[page,row,col] = 0.0
orbits = orbits.reshape((1,orbits.size))

if kick_reg == 1:
# Load kick model output
    print('---------- Read kick matrix ----------')
    # filename = 'C:/Users/larsen/Documents/Analytical orbit tomography/Resonant mode-particle interactions/kick matrices/pDEDP_a5p0.AEP' # This seems wrong!
    filename = 'C:/Users/larsen/Documents/Analytical orbit tomography/Resonant mode-particle interactions/kick matrices/pDEDP_all_a5.AEP'
    # filename = 'C:/Users/larsen/Documents/Analytical orbit tomography/Resonant mode-particle interactions/kick matrices/sparse_pDEDP.AEP' # This seems very wrong!
    kick_dict = read_kick_matrix(filename, ENBI)

    # Interpolate kick matrix onto inversion phase-space grid
    print('---------- Interpolate kick matrix onto inversion grid ----------')
    kick_dict_new = interpolate_kick_matrix(kick_dict, Eaxis, Laxis, Paxis, ENBI)
    del kick_dict
    
    # Plot kicks
    # plot_kicks(kick_dict_new, ENBI)

    # Make kick-model-regularisation matrix
    print('---------- Make kick-model-regularisation matrix ----------')
    print('With sparse matrices, this does not use too much memory or CPU resources!\nSparse matrices is allowed in CUQIpy.')
    h = h5py.File("C:/Users/larsen/Documents/Analytical orbit tomography/Resonant mode-particle interactions/Phase-space weights/86327/time_0-48/CoMweights_highres.h5", 'r')
    orbit_types = h['orbits'][()].astype(str)
    h.close()
    L = make_kick_regularisation_matrix(kick_dict_new, ENBI, orbit_types, Gazzola)
    # plot_precision_matrix(L, Eaxis, Laxis, Paxis)
    if Gazzola == 0:
        print('---------- Make also L1 regularisation matrix ----------')
        L1extra = make_L1_regularisation_matrix(Eaxis, Laxis, Paxis, ENBI)
    # If using zeroth-order Tikhonov in addition to kick prior:
    # L = scipy.sparse.csr_matrix(scipy.sparse.vstack([L, scipy.sparse.eye(L.shape[1])])) # Try putting zeroth-order Tikhonov everywhere underneath kick prior
    del kick_dict_new

elif L1 == 1:
    L = make_L1_regularisation_matrix(Eaxis, Laxis, Paxis, ENBI)
    # plot_precision_matrix(L, Eaxis, Laxis, Paxis)
else:
    L = None

# Prepare to remove cold region etc.
# If including only system 5 and 15, i.e. removing system 19
if len(sys_list) == 2:
    W = W[:(240*40),:]
# Load FIDASIM spectra
h = h5py.File("C:/Users/larsen/Documents/Analytical orbit tomography/Resonant mode-particle interactions/FIDASIM spectra/86327/time_1-05/9013253584_spectra.h5", 'r')
lambda_axis = h['lambda'][()]
h.close()
# Remove cold region and Carbon-II lines
idx_start = np.argmin(np.abs(lambda_axis - 652.0)) # Remove everything before 652.0 nm. No fast-ions there anyway
idx_end   = np.argmin(np.abs(lambda_axis - 660.0)) # Remove everything after 660.0 nm. No fast-ions there anyway
idx_cold1 = np.argmin(np.abs(lambda_axis - 654.7)) # Approximately just before cold contributions
idx_cii2  = np.argmin(np.abs(lambda_axis - 658.5)) # Index right after Carbon II lines
# Try not to cut CII lines away, because we want to see recons in ctr-passing region
# if co_going == False:
if include_CII_spectra:
    idx_cii2  = np.argmin(np.abs(lambda_axis - 656.1))
for ch_num in range(len(sys_list)*len(ch_list)):
    print('channel number index: ', ch_num)
    idx1 = ch_num*len(lambda_axis) + idx_start - 1
    idx2 = ch_num*len(lambda_axis) + idx_cold1
    idx3 = ch_num*len(lambda_axis) + idx_cii2 - 1
    idx4 = ch_num*len(lambda_axis) + idx_end + 1
    arr1 = np.linspace(ch_num*len(lambda_axis), idx1, idx1-ch_num*len(lambda_axis)+1).astype(int)
    arr2 = np.linspace(idx2, idx3, idx3-idx2+1).astype(int)
    arr3 = np.linspace(idx4, (ch_num+1)*len(lambda_axis)-1, (ch_num+1)*len(lambda_axis)-1-idx4+1).astype(int)
    try:
        arr = np.hstack((arr, arr1, arr2, arr3))
    except NameError:
        arr = np.hstack((arr1, arr2, arr3))
    print('rows removed: ', len(arr1)+len(arr2)+len(arr3))
W = np.delete(W, arr, axis=0)
# Only use 40 LOS
W = W[:int(40*W.shape[0]//60), :]
# Apply Gaussian filter on weight functions (ATTEMPT!)
# print('Applying Gaussian filter on weight functions (ATTEMPT!)')
# from scipy.ndimage import gaussian_filter
# for row in range(W.shape[0]):
#     wtemp = W[row, :].reshape((len(Eaxis), len(Laxis), len(Paxis)))
#     wtemp = gaussian_filter(wtemp, sigma=0.8)
#     W[row, :] = wtemp.reshape((1, W.shape[1]))
# print('Gaussian filter applied')


if (TRANSP == 0) & (ASCOT == 0):
    # Make Gaussian blob as test distribution [energy [kev], Lambda [-], Pphi [-]]
    print('---------- Make test distribution ----------')
    mean = [18.0, 0.5, 0.0] # Center of blob in each direction
    std = [1.0, 0.1, 0.1] # Standard deviation of blob in each direction
    dist = make_test_distribution(Eaxis, Laxis, Paxis, mean, std)
    # Make phantoms around phase space
    dist = np.zeros(dist.shape)
    LL = [0.2, 0.95]
    PP = [0.55, 0.1]
    Lneg = [0.2, 0.5]
    # # Only one blob
    # LL = [0.2]
    # Lneg = [0.35]
    for e, E in enumerate(Eaxis):
        if E > 0:
            PP = -np.array([(1.2-0.1) * (E-5.0)/(27.7-5.0) + 0.1, (0.1+0.3) * (E-5.0)/(27.7-5.0) - 0.3])
            for l, Li in enumerate(LL):
                mean = [E, Li, PP[l]] # Center of blob in each direction
                std = [1.0, 0.05, 0.2] # Standard deviation of blob in each direction
                dist += make_test_distribution(Eaxis, Laxis, Paxis, mean, std) * 1e15
        else:
            if len(Lneg) > 1:
                Pneg = -np.array([(-1.75+1.2) * (np.abs(E)-7.3)/(30.0-7.3) - 1.2, (-1.35+0.9) * (np.abs(E)-7.3)/(30.0-7.3) - 0.9])
            else:
                Pneg = -np.array([(-1.50+1.1) * (np.abs(E)-7.3)/(30.0-7.3) - 1.1])
            for i in range(len(Lneg)):
                mean = [E, Lneg[i], Pneg[i]] # Center of blob in each direction
                std = [1.0, 0.05, 0.15] # Standard deviation of blob in each direction
                dist += make_test_distribution(Eaxis, Laxis, Paxis, mean, std, rot=1) * 1e15

elif ASCOT == 1:
    # Use ASCOT distribution
    print('---------- Use ASCOT distribution ----------')
    h = h5py.File("C:/Users/larsen/Documents/Analytical orbit tomography/Resonant mode-particle interactions/ASCOT distributions/86327/time_1-05/9013253584_distribution_CoM.h5", 'r')
    dist = h['dist'][()]
    h.close()
    # Load FIDASIM spectra
    h = h5py.File("C:/Users/larsen/Documents/Analytical orbit tomography/Resonant mode-particle interactions/FIDASIM spectra/86327/time_1-05/9013253584_spectra.h5", 'r')
    s_array = np.empty(shape=(40,h['fida'].shape[1]))
    for ch in range(40):
            s_array[ch,:] = h['fida'][ch,:]
    if experimental == 1:
        print('DO THIS!')
    # Rescale weight functions to match FIDASIM spectra
    print('---------- Rescaling weight functions ----------')
    s_ASCOT = W @ dist.reshape((W.shape[1]))
    # Index of 654nm for rescaling
    idx654 = np.argmin(np.abs(np.array(lambda_axis) - 654.0))
    for ch_num in range(len(sys_list)*len(ch_list)):
        idx = ch_num*len(lambda_axis)
        # Rescale the spectra from the weight functions.
        # Rescaling with the maximum measurement is not the best choice, as there can be weird spikes
        # This is for examples seen in several channels of system 15.
        # Assuming the weird spikes happen in the cold region, we can rescale with the maximum outside of that.
        # For example rescale with the maximum below 654nm or something like that.
        W[idx:(idx+len(lambda_axis)),:] = W[idx:(idx+len(lambda_axis)),:] * np.max(s_array[ch_num,:(idx654+1)]) / np.max(s_ASCOT[idx:(idx+len(lambda_axis))])
    del s_ASCOT
    # Remove cold region and Carbon-II lines
    h.close()
    s_temp = s_array[:,idx_start:idx_cold1]
    s_active = np.hstack((s_temp, s_array[:,idx_cii2:(idx_end+1)]))
    l_temp = lambda_axis[idx_start:idx_cold1]
    lambda_active = np.hstack((l_temp, lambda_axis[idx_cii2:(idx_end+1)]))
    del s_temp, l_temp
    s = s_active.reshape((s_active.size, 1))
    del s_active
    
elif TRANSP == 1:
    
    print('---------- Load TRANSP CoM space distribution ----------')
    if co_going:
        foldername = 'C:/Users/larsen/Documents/Analytical orbit tomography/Resonant mode-particle interactions/TRANSP fast-ion distributions/'
        filename = f'fi_co_{kicktime}_CoM.h5' # Number 2 is after kick. Number 5 is at 0.48s, so just before kick, I guess. I guess the numbering is (1,2,3,4,5,6) before and after three fisbone cycles. But the kick matrices are at time 0.48s.
        transpfile = f'86327P39_fi_{kicktime}.cdf'
    else:
        foldername = 'C:/Users/larsen/Documents/Analytical orbit tomography/Resonant mode-particle interactions/TRANSP fast-ion distributions/With_NBI2_ctr-going/'
        filename = f'fi_ctr_{kicktime}_CoM.h5' # Number 2 is after kick. Number 5 is at 0.48s, so just before kick, I guess. I guess the numbering is (1,2,3,4,5,6) before and after three fisbone cycles. But the kick matrices are at time 0.48s.
        transpfile = f'86327P55_fi_{kicktime}.cdf'
    h = h5py.File(foldername+filename, 'r')
    dist = h['dist'][()]
    h.close()
    
    FIdist, R2D, Z2D, ETRANSP, pTRANSP, time = read_TRANSP_cdf(foldername, transpfile)

if ASCOT == 0:
    if co_going == True:
        # Remove everything but co-going and trapped orbits
        print('---------- Removing everything but co-going and trapped orbits ----------')
        W = W[:,(W.shape[1] // 2):]
        if L is not None:
            L = L[:,(L.shape[1] // 2):]
        orbits = orbits[:, (orbits.size // 2):]
        dist = dist[(dist.shape[0] // 2):, :, :]
    elif not co_going:
        # Remove everything but counter-passing orbits
        print('---------- Removing everything but counter-passing orbits ----------')
        W = W[:,:(W.shape[1] // 2)]
        if L is not None:
            L = L[:,:(L.shape[1] // 2)]
        orbits = orbits[:, :(orbits.size // 2)]
        dist = dist[:(dist.shape[0] // 2), :, :]
    elif co_going == 'projection':
        print('---------- Keeping all orbits ----------')
    
    # Synthetic spectra
    print('---------- Compute synthetic spectra ----------')
    s_clean = W @ dist.reshape((W.shape[1]))
    # Add noise
    print('---------- Add noise to synthetic spectra ----------')
    noise = np.random.normal(loc=0.0, scale=0.01*np.max(s_clean), size=s_clean.size)
    noise = np.random.normal(loc=0.0, scale=0.10*s_clean, size=s_clean.size)
    noisefloor = np.ones(noise.size) * 1e-6 * np.max(s_clean)
    if co_going == False:
        try:
            nn = np.loadtxt('noise_for_tomo_tests_ctr.txt')
            s = s_clean + nn
            print('---------- Using same noise realisation for fair testing ----------')
        except FileNotFoundError:
            s = s_clean + np.max(np.vstack((noise, noisefloor)), axis=0)
            np.savetxt('noise_for_tomo_tests_ctr.txt', np.max(np.vstack((noise, noisefloor)), axis=0))
    else:
        try:
            nn = np.loadtxt('noise_for_tomo_tests_co_woutCII.txt')
            s = s_clean + nn
            print('---------- Using same noise realisation for fair testing ----------')
        except FileNotFoundError:
            s = s_clean + np.max(np.vstack((noise, noisefloor)), axis=0)
            np.savetxt('noise_for_tomo_tests_co_woutCII.txt', np.max(np.vstack((noise, noisefloor)), axis=0))
    
if (kick_reg == 1) & (Gazzola == 0):
    L1extra = L1extra[:,:(L1extra.shape[1] // 2)]
# Remove invalid region of phase space from problem
print('---------- Removing invalid region of phase space ----------')
dist_for_proj = (dist.reshape((W.shape[1])))[np.where(orbits[0] > 0)[0]]
W = W[:,np.where(orbits[0] > 0)[0]]
if L is not None:
    # prec = (L.T)@L # Doesn't work either
    # prec = prec[:,np.where(orbits[0] > 0)[0]]
    # prec = prec[np.where(orbits[0] > 0)[0],:]
    L = L[:,np.where(orbits[0] > 0)[0]] # This makes it not positive semi-definite apparently...
    if (kick_reg == 1) & (Gazzola == 0):
        L1extra = L1extra[:,np.where(orbits[0] > 0)[0]]
    # ll1 = L[:orbits.size,:]
    # ll2 = L[orbits.size:(2*orbits.size),:]
    # if L.shape[0] > 3*dist.size:
    #     ll3 = L[(2*orbits.size):(3*orbits.size),:]
    #     ll4 = L[(3*orbits.size):,:]
    #     L = scipy.sparse.vstack([ll1[np.where(orbits[0] > 0)[0],:],
    #                     ll2[np.where(orbits[0] > 0)[0],:],
    #                     ll3[np.where(orbits[0] > 0)[0],:],
    #                     ll4[np.where(orbits[0] > 0)[0],:]]) # This is not enough to keep it positive semi-definite apparently...
    #     del ll4
    # else:
    #     ll3 = L[(2*orbits.size):,:]
    #     L = scipy.sparse.vstack([ll1[np.where(orbits[0] > 0)[0],:],
    #                     ll2[np.where(orbits[0] > 0)[0],:],
    #                     ll3[np.where(orbits[0] > 0)[0],:]]) # This is not enough to keep it positive semi-definite apparently...
    # del ll1, ll2, ll3
del orbits
print('W shape:', W.shape)

# ========================================================================================================
# Project true distribution onto right singular vectors of weight matrix.
# The is the very best we can ever hope to reconstruct with Tikhonov regularisation.
print('---------- Computing singular value decomposition of weight matrix ----------')
U, s_vals, Vt = np.linalg.svd(W) # Vt = V^T is the transpose of V
del U

# Plot singular values
if ASCOT == 1:
    cut_idx = 3400
else:
    cut_idx = 2000
plt.figure()
plt.plot(s_vals, 'k-')
plt.yscale('log')
plt.plot(cut_idx * np.ones((10,1)), np.logspace(-14, 6, 10), 'k--')
plt.grid(True)
plt.xlabel('Index $i$ [-]', fontsize=14)
plt.ylabel('Singular value $\Sigma_{ii}$ [-]', fontsize=14)
if ASCOT == 1:
    plt.ylim(1e-8, 1e3)
else:
    plt.ylim(1e-12, 1e5)

# Plot expansion coefficients
plt.figure()
plt.plot(np.abs(Vt@dist_for_proj), 'k-')
plt.yscale('log')
plt.plot(cut_idx * np.ones((10,1)), np.logspace(11, 18, 10), 'k--')
plt.grid(True)
plt.xlabel('Index $i$ [-]', fontsize=14)
plt.ylabel('Expansion coefficient $\\alpha_{i}$ [-]', fontsize=14)
if ASCOT == 1:
    plt.ylim(1e-8, 1e3)
else:
    plt.ylim(1e12, 1e17)
# Plot projections
ftemp = (Vt.T)@Vt@dist_for_proj
fproj_full = np.zeros((len(Eaxis)*len(Laxis)*len(Paxis), 1))
fproj_full[indices] = ftemp.reshape((ftemp.size, 1))
del ftemp
ftemp = (Vt[:cut_idx,:].T)@Vt[:cut_idx,:]@dist_for_proj
fproj_cut = np.zeros((len(Eaxis)*len(Laxis)*len(Paxis), 1))
fproj_cut[indices] = ftemp.reshape((ftemp.size, 1))

# Plot projection
plot_3D_inversions(fproj_full, dist, Eaxis, Laxis, Paxis)
plot_3D_inversions(fproj_cut, dist, Eaxis, Laxis, Paxis)
# ========================================================================================================

# Solve inverse problem
print('---------- Solve inverse problem ----------')
if (cuqipy == 0) & (hmc == 0):

    inversions = tikhonov_3D_reg(s, W, alpha, L)

elif cuqipy == 1:

    print('\n---------- Bayesian inference with CUQIpy ----------')
    
    # Make forward model
    A = LinearModel(W)
    # Make hyperprior
    spara = cuqi.distribution.Gamma(shape=1e1, rate=1e1)
    # Make prior distribution
    if L is None:
        x = NonnegativeGaussian(mean=np.zeros(W.shape[1]), prec=1e-1, geometry=A.domain_geometry)
    elif L1 == 1:
        prec = ((L.T)@L).todense() # I have to make it to dense array. Otherwise CUQIpy tells me it is not positive semi-definite...
        x = NonnegativeGaussian(mean=np.zeros(W.shape[1]), prec=1e-4 * prec, geometry=A.domain_geometry)
    else:
        if Gazzola == 0:
            prec = ((L.T)@L).todense() + ((L1extra.T)@L1extra).todense() # I have to make it to dense array. Otherwise CUQIpy tells me it is not positive semi-definite...
            del L1extra
        elif Gazzola == 1:
            prec = ((L.T)@L).todense()
        if hyperparam & Gazzola:
            x = NonnegativeGaussian(mean=np.zeros(W.shape[1]), prec=lambda spara: 1e-4 * precision_matrix(spara[-1], L), geometry=A.domain_geometry)
        else:
            x = NonnegativeGaussian(mean=np.zeros(W.shape[1]), prec=1e-4 * prec, geometry=A.domain_geometry)
        
    # Generate a CUQI array with the observed data s in "s = Wf"
    y_obs = cuqi.array.CUQIarray(s.flatten(), geometry=cuqi.geometry.Discrete(A.range_dim))
    # Likelihood of measurements s
    Cs = np.sqrt(s).reshape((len(s), 1))
    y = cuqi.distribution.Gaussian(mean=A@x, cov=np.diagonal(Cs), geometry=A.range_geometry)
    # Make joint distribution
    if hyperparam:
        joint = cuqi.distribution.JointDistribution(spara, x, y)
    else:
        joint = cuqi.distribution.JointDistribution(x, y)
    # Make posterior distribution by conditioning the joint distribution on the observed data
    posterior = joint(y=y_obs)
    # # Make Bayesian problem
    # BP = BayesianProblem(x, y)
    # BP.set_data(y=y_obs)
    # MAPest = BP.MAP() # DOESN'T GIVE GOOD RESULTS
    # UQest = BP.UQ()
    # Make sampling strategy
    if hyperparam:
        sparsity_null = lambda z : count_nonzero(z, threshold = 1e-6)
        sampling_strategy = {'spara': CustomConjugatePair(sparsity_null),
                             'x': RegularizedLinearRTO(initial_point=np.zeros(W.shape[1]), maxit=100, penalty_parameter=1e0)}
        num_sampling_steps = {'spara': 1, 'x': 1}
    else:
        sampling_strategy = {'x': RegularizedLinearRTO(initial_point=np.zeros(W.shape[1]), maxit=100, penalty_parameter=1e0)}
        num_sampling_steps = {'x': 1}
    if hyperparam:
        sampler = HybridGibbs(target=posterior, sampling_strategy=sampling_strategy, num_sampling_steps=num_sampling_steps)
    else:
        sampler = RegularizedLinearRTO(target=posterior, initial_point=np.zeros(W.shape[1]), maxit=100, penalty_parameter=1e0)
    sampler.warmup(nburn)
    sampler.sample(nsamples)
    
    # Sample posterior distribution as a hierarchical Bayesian model
    posterior_samples = sampler.get_samples()
    post_mean = np.mean(posterior_samples.samples[:,nburn:], axis=1)
    post_mean = post_mean.reshape((post_mean.size, 1))
    post_mean_size = post_mean.size
    
    # Compute metrics before adding the pixels corresponding to invalid orbits etc.
    rmse  = np.sqrt(mean_squared_error(dist_for_proj.reshape((post_mean.shape[0])), post_mean))
    nrmse = rmse / np.linalg.norm(dist_for_proj)
    ssims  = ssim(post_mean.flatten(), dist_for_proj.reshape((post_mean.shape[0])), data_range=np.max(dist_for_proj) - np.min(dist_for_proj)) # SSIM is symmetric in the images.
    psnrs  = PSNR(dist_for_proj.reshape((post_mean.shape)), post_mean)
    print('nRMSE =', nrmse)
    print('PSNR =', psnrs)
    print('SSIM =', ssims)
    
    inversions = np.zeros((len(Eaxis)*len(Laxis)*len(Paxis), 1))
    if co_going:
        inversions[indices[(len(indices)-post_mean_size):]] = post_mean
    else:
        inversions[indices[:post_mean_size]] = post_mean
    del post_mean
    post_std = np.std(posterior_samples.samples[:,nburn:], axis=1)
    post_std = post_std.reshape((post_std.size, 1))
    uncertainties = np.zeros((len(Eaxis)*len(Laxis)*len(Paxis), 1))
    if co_going:
        uncertainties[indices[(len(indices)-post_std.size):]] = post_std
    else:
        uncertainties[indices[:post_std.size]] = post_std
    del post_std
    # Plot uncertainties
    # uncertainties = np.nan_to_num(uncertainties/inversions)
    plot_3D_inversions(uncertainties, dist, Eaxis, Laxis, Paxis, unc=1)
    
elif hmc == 1:
    
    print('\n---------- Bayesian inference with hamiltorch ----------')
    import torch
    import hamiltorch
    
    dim = W.shape[1]
    snorm = np.linalg.norm(s)
    Wnorm = np.linalg.norm(W)
    s = s / snorm
    W = W / Wnorm
    
    def log_prob_func(x, dim=dim):
        
        # Likelihood
        likelihood_mean = torch.tensor(s)
        likelihood_prec = torch.eye(s.size)
        values = torch.matmul(torch.tensor(W, dtype=torch.float32), x)
        loglikelihood = torch.distributions.multivariate_normal.MultivariateNormal(loc=likelihood_mean, precision_matrix=likelihood_prec).log_prob(values).sum()
        # Prior
        prior_mean = torch.zeros(dim)
        prior_prec = torch.eye(dim)
        logprior = torch.distributions.multivariate_normal.MultivariateNormal(loc=prior_mean, precision_matrix=prior_prec).log_prob(x).sum()
        
        return loglikelihood + logprior
    
    def hmc_sample(dim):
        
        num_samples = 1000
        step_size = .3
        params_init = torch.ones(dim)
        num_steps_per_sample = 5
        
        x_hmc = hamiltorch.sample(log_prob_func=log_prob_func, params_init=params_init, num_samples=num_samples, step_size=step_size, num_steps_per_sample=num_steps_per_sample)
        
        return x_hmc

    samples = np.array(hmc_sample(dim))
    post_mean = np.mean(samples, axis=0)

    plt.figure()
    plt.imshow(post_mean.reshape((20,20)))
    plt.colorbar()

# Plot inversions
plot_3D_inversions(inversions, dist, Eaxis, Laxis, Paxis)

# Plot synthetic spectra from inversions
rmses  = np.zeros(inversions.shape[1])
nrmses = np.zeros(inversions.shape[1])
ssims  = np.zeros(inversions.shape[1])
psnrs  = np.zeros(inversions.shape[1])

temp = W
W = np.zeros((W.shape[0],inversions.size))
fig, ax = plt.subplots()
plt.plot(s, 'b')
if co_going:
    W[:,indices[(len(indices)-post_mean_size):]] = temp
else:
    W[:,indices[:post_mean_size]] = temp
del temp
if dist.shape[0] == Eaxis.size//2:
    if co_going == False:
        dist = np.concatenate((dist, np.zeros((dist.shape[0], dist.shape[1], dist.shape[2]))), axis=0)
    elif co_going == True:
        dist = np.concatenate((np.zeros((dist.shape[0], dist.shape[1], dist.shape[2])), dist), axis=0)
for i in range(inversions.shape[1]):
    plt.plot(W@inversions[:,i], '--r')
    rmses[i]  = np.sqrt(mean_squared_error(dist.reshape((inversions.shape[0])), inversions[:,i]))
    nrmses[i] = rmses[i] / np.linalg.norm(dist)
    ssims[i]  = ssim(dist.reshape((inversions.shape[0])), inversions[:,i], data_range=np.max(dist) - np.min(dist))
    psnrs[i]  = PSNR(dist.reshape((inversions.shape[0])), inversions[:,i])
ax.set_ylabel('Signal intensity', fontsize=14)
ax.set_xlabel('Measurement bin index', fontsize=14)
ax.set_title('Systems 5 and 15, all channels', fontsize=14)
ax.legend(['FIDASIM spectra', 'Reconstructed spectra'], fontsize=14)
plt.ion()

print('\nCHECK WEIGHT FUNCTIONS AGAIN, ESPECIALLY IN CTR-PASSING!!!')

# # Plot precision matrix
# print('---------- Plot precision matrix ----------')
# temp = L
# L = np.zeros((L.shape[0], W.shape[1]))
# L[:, indices] = temp.todense()
# del temp
# precision = (L.T)@L
# Eidx = np.argmin(np.abs(-25.0 - Eaxis))
# Lidx = np.argmin(np.abs(0.25 - Laxis))
# Pidx = np.argmin(np.abs(1.80 - Paxis))
# lin_idx = np.ravel_multi_index((Eidx, Lidx, Pidx), dist.shape)
# prec = precision[:,lin_idx]
# # prec = np.sum(np.abs(precision), axis=1)
# plot_3D_inversions(prec.reshape((prec.size, 1)), dist, Eaxis, Laxis, Paxis, prec=1)
    