import numpy as np

# Function to read kick matrix calculated by Orbit for a given TCV shot.
# The kick matrix is written in sparse representation.
# Output: 2D array where each column denotes the value of (\Delta E, \Delta Pphi, Eidx, Pphiidx, Muidx, kick probability)

def read_kick_matrix(filename, ENBI):

    f = open(filename, 'r')

    # Get the headlines of the ASCII file out of the way
    Headlines = []
    for i in range(12):
        Headlines.append(f.readline())

    # Continue reading the file to get the length of each dimension
    axes_dims = []
    for i in range(5):
        line = f.readline()
        line = line.strip()
        columns = line.split()
        axes_dims.append(int(line.split()[0]))

    # The following 23 lines define the DE, DPphi, E, Pphi and mu axes.
    # I must figure out how to split them up properly.
    # I should be able to do this with 'axes_dims'
    combined_axes = []
    for i in range(23):
        line = f.readline()
        line = line.strip()
        axes = line.split()
        combined_axes += [float(a) for a in axes]

    # Split arrays properly
    dEaxis    = combined_axes[0:axes_dims[0]]
    dPphiaxis = combined_axes[axes_dims[0]:sum(axes_dims[0:2])]
    Eaxis      = combined_axes[sum(axes_dims[0:2]):sum(axes_dims[0:3])]
    Pphiaxis   = combined_axes[sum(axes_dims[0:3]):sum(axes_dims[0:4])]
    muaxis     = combined_axes[sum(axes_dims[0:4]):sum(axes_dims[0:5])]

    # Read the rest of the file to obtain the probability value at each phase-space point
    kick_prob_array_init = []
    for line in f:
        line = line.strip()
        columns = line.split() # Maybe I need the last column anyway... Don't need the last column, which is some MC counter or something...
        kick_prob_array_init.append([float(c) for c in columns])
        # The strength of the kick should be sqrt(dE^2 + dPphi^2)

    kick_prob_array_init = np.array(kick_prob_array_init)

    # Find (E,mu,Pphi) triplets with multiple kicks
    triplets, indices, inv_indices, counts = np.unique(kick_prob_array_init[:,2:5], axis=0, return_index=True, return_inverse=True, return_counts=True)
    print("--------------- Number of triplets with more than one kick: ", triplets[counts>1].shape[0])
    print("--------------- Making weighted average ---------------")
    # To deal with this in the regularisation, we probably need to make a weighted average or something...
    kick_prob_array = np.empty((indices.size,kick_prob_array_init.shape[1]))
    dEarray = np.array(dEaxis) / ENBI # I think I should normalise this to injection energy such that all axes are of dimension 1. I think it is still compatible with "make_kick_regularisation_matrix"
    dParray = np.array(dPphiaxis)
    for i, idx in enumerate(indices):
        idxs = np.where(inv_indices == i)[0]
        probs = kick_prob_array_init[idxs, 5]/np.sum(kick_prob_array_init[idxs, 5]) # Probability of each kick
        # Weight each dE, dPphi kick with the probability that they occur
        dE    = probs * dEarray[(kick_prob_array_init[idxs, 0]-1).astype(int)]
        dPphi = probs * dParray[(kick_prob_array_init[idxs, 1]-1).astype(int)]
        # Sum up all the kicks weighted by proabability of occurring
        weighted_average = np.sum(np.array([dE, dPphi]).T, axis=0)
        #weighted_average = np.mean(np.array([dE, dPphi]).T, axis=0)
        kick_prob_array[i,0] = np.argmin(np.abs(dEarray - weighted_average[0])) + 1 # dEindex in "Matlab counting"
        kick_prob_array[i,1] = np.argmin(np.abs(dParray - weighted_average[1])) + 1 # dPphiindex in "Matlab counting"
        kick_prob_array[i,2:5]  = triplets[i,:] # in "Matlab counting"
        # Last column is the strength of the effective kick
        # This one does not agree with the one below! kick_prob_array[i,5] = np.sqrt(weighted_average[0]**2 + weighted_average[1]**2)
        # It's probably because the weighted average might not be close to any grid points in dEaxis or dPphiaxis!
        #kick_prob_array[i,5] = np.sqrt(dEarray[int(kick_prob_array[i,0])-1]**2 + dParray[int(kick_prob_array[i,1])-1]**2)
        # The kick strength as computed above leaves a white band between the blue and red kicks, because the multiple kicks cancel each other out.
        # Rather, I want to smooth along the kick direction inside this white band too. So compute it as the root mean square of all the individual kicks. That is what Mario does in his plot.
        #kick_prob_array[i,5] = np.sqrt(weighted_average[0]**2 + weighted_average[1]**2)
        # Alternative, without 'probs':
        kick_prob_array[i,5] = np.sqrt(np.mean(dEarray[(kick_prob_array_init[idxs, 0]-1).astype(int)]**2)+np.mean(dParray[(kick_prob_array_init[idxs, 1]-1).astype(int)]**2))
        # Alternative, define direction of kick, without 'probs':
        # Direction defined by root mean square (this can mess up the direction)
        sE = np.sign(np.mean(dEarray[(kick_prob_array_init[idxs, 0]-1).astype(int)]))
        sP = np.sign(np.mean(dParray[(kick_prob_array_init[idxs, 1]-1).astype(int)]))
        # sE = 1
        # sP = 1
        dEdir = sE*np.sqrt(np.mean(dEarray[(kick_prob_array_init[idxs, 0]-1).astype(int)]**2))
        dPphidir = sP*np.sqrt(np.mean(dParray[(kick_prob_array_init[idxs, 1]-1).astype(int)]**2))
        # Direction defined by mean (can end up with zero kick)
        # dEdir = np.mean(dEarray[(kick_prob_array_init[idxs, 0]-1).astype(int)])
        # dPphidir = np.mean(dParray[(kick_prob_array_init[idxs, 1]-1).astype(int)])
        kick_prob_array[i,0] = np.argmin(np.abs(dEarray - dEdir)) + 1
        kick_prob_array[i,1] = np.argmin(np.abs(dParray - dPphidir)) + 1

    # Add a column with the kick strength = sqrt(dE^2 + dPphi^2)
    kick_strength = kick_prob_array[:,5]
    
    # Use np.add.at(kick_prob_array, indices, kick_prob_array[:,5])
    E3D, _, _ = np.meshgrid(Eaxis, muaxis, Pphiaxis, indexing='ij')
    kick_phase_space = np.zeros(E3D.shape)

    Eidxs = list(map(int, kick_prob_array[:,2]-1))
    Pidxs = list(map(int, kick_prob_array[:,3]-1))
    midxs = list(map(int, kick_prob_array[:,4]-1))
    np.add.at(kick_phase_space, (Eidxs, midxs, Pidxs), kick_strength)

    # Make dE kick in phase space
    dE_kick_phase_space = np.zeros(E3D.shape)
    np.add.at(dE_kick_phase_space, (Eidxs, midxs, Pidxs), dEarray[(kick_prob_array[:,0]-1).astype(int)])

    # Make dPphi kick in phase space
    dPphi_kick_phase_space = np.zeros(E3D.shape)
    np.add.at(dPphi_kick_phase_space, (Eidxs, midxs, Pidxs), dParray[(kick_prob_array[:,1]-1).astype(int)])

    kick_dict = {'kick_prob_array': kick_prob_array, 'kick_phase_space': kick_phase_space, 
                    'dE_kick_phase_space': dE_kick_phase_space, 'dPphi_kick_phase_space': dPphi_kick_phase_space, 
                    'dEaxis': np.array(dEaxis), 'dPphiaxis': np.array(dPphiaxis),
                    'Eaxis': np.array(Eaxis), 'Pphiaxis': np.array(Pphiaxis), 'muaxis': np.array(muaxis)}
    
    return kick_dict

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


def rotation_matrix(angle=np.pi, axis=0):
    
    if axis == 0:
        rot_mat = np.array([[1, 0, 0],[0, np.cos(angle), -np.sin(angle)],[0, np.sin(angle), np.cos(angle)]])
    elif axis == 1:
        rot_mat = np.array([[np.sin(angle), 0, np.cos(angle)],[0, 1, 0],[np.cos(angle), 0, -np.sin(angle)]])
    elif axis == 2:
        rot_mat = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
        
    return rot_mat

# ========================================================================================

# Function to reorganise the kick matrix into a regularisation matrix for inversions
# !!! I should change the mu-direction to the lambda direction. Zero change in mu implies change in Lambda!!!

# Inputs needed:
#   - Kick matrix as returned by the read_kick_matrix(filename) function.
#   - Energy axis of phase space in units of keV
#   - Pphi axis of phase space in normalised units
#   - Lambda axis of phase space in normalised units (obviously)

# Output:
#   - L, regularisation matrix of dimension (3M x M), where M is len(Eaxis)*len(Pphiaxis)*len(Lambdaaxis)

def make_kick_regularisation_matrix(kick_dict, ENBI, orbit_types, Gazzola=0):
    
    import h5py

    # Unpack dictionary
    kick_matrix = kick_dict['kick_prob_array']
    kick_strength          = kick_dict['kick_phase_space'].flatten() # 3D array, with the kick strength in each point in (E,Pphi,mu) phase space
    dE_kick_phase_space    = kick_dict['dE_kick_phase_space'].flatten()
    dPphi_kick_phase_space = kick_dict['dPphi_kick_phase_space'].flatten()
    dEaxis      = kick_dict['dEaxis']
    dPphiaxis   = kick_dict['dPphiaxis']
    Eaxis       = kick_dict['Eaxis']
    Pphiaxis    = kick_dict['Pphiaxis']
    muaxis      = kick_dict['muaxis']
    shape = kick_dict['kick_phase_space'].shape
    # Grid spacing
    deltaE = np.mean(np.diff(Eaxis[len(Eaxis)//2:])) / ENBI # Normalise with NBI energy such that all dimensions are of order 1.
    deltaPphi = np.mean(np.diff(Pphiaxis))
    deltamu = np.mean(np.diff(muaxis))
    # Try this. Then it basically only penalises in the energy direction
    # deltaE = 1/ENBI # Then it basically only penalises in the energy direction
    # For now, these below are what I use
    #deltaE = 1
    #deltaPphi = 1
    #deltamu = 1
    
    # Unpack various variables dict
    various_variables_dict = h5py.File('C:/Users/larsen/Documents/Analytical orbit tomography/Resonant mode-particle interactions/Tomography/FIDA/86327/Time_0-48/various_variables_dict_86327_0.480.h5', 'r')
    Qe = various_variables_dict['q'][()]
    mD = various_variables_dict['m'][()]
    psiwall = various_variables_dict['psiwall'][()]
    B0 = various_variables_dict['B0'][()]
    B_LFS = various_variables_dict['B_LFS'][()]
    Bphi_LFS = various_variables_dict['Bphi_LFS'][()]
    R = various_variables_dict['R'][()]
    R_LFS = various_variables_dict['R_LFS'][()]
    Z_LFS = various_variables_dict['Z_LFS'][()]
    Rvec = various_variables_dict['Rvec'][()]
    zvec = various_variables_dict['zvec'][()]
    pitch = various_variables_dict['pitch'][()]
    various_variables_dict.close()
    R, Z = np.meshgrid(Rvec, zvec)
    curdir = -1
    various_variables_dict.close()

    # Make first-order gradient operator
    import scipy
    # LE = scipy.sparse.diags([-1,1],[0,1], shape=(len(Eaxis),len(Eaxis))).todense()
    # LP = scipy.sparse.diags([-1,1],[0,1], shape=(len(Pphiaxis),len(Pphiaxis))).todense()
    # Lm = scipy.sparse.diags([-1,1],[0,1], shape=(len(muaxis),len(muaxis))).todense()
    # If sparse matrix
    LE = scipy.sparse.csr_matrix(scipy.sparse.diags([-1,1],[0,1], shape=(len(Eaxis),len(Eaxis)))) / deltaE
    LP = scipy.sparse.csr_matrix(scipy.sparse.diags([-1,1],[0,1], shape=(len(Pphiaxis),len(Pphiaxis)))) / deltaPphi
    Lm = scipy.sparse.csr_matrix(scipy.sparse.diags([-1,1],[0,1], shape=(len(muaxis),len(muaxis)))) / deltamu
    ''' # The out-commented stuff here is wrong! Why? This looks correct!
    L1 = np.concatenate((np.kron(LP,np.kron(np.eye(len(muaxis)),np.eye(len(Eaxis)))),
                         np.kron(np.eye(len(Pphiaxis)),np.kron(Lm,np.eye(len(Eaxis)))),
                         np.kron(np.eye(len(Pphiaxis)),np.kron(np.eye(len(muaxis)),LE))), axis=0)
    '''
    #L1 = np.concatenate((np.kron(LE,np.kron(np.eye(len(muaxis)),np.eye(len(Pphiaxis)))),
    #                     np.kron(np.eye(len(Eaxis)),np.kron(Lm,np.eye(len(Pphiaxis)))),
    #                     np.kron(np.eye(len(Eaxis)),np.kron(np.eye(len(muaxis)),LP))), axis=0)
    L1 = scipy.sparse.csr_matrix(scipy.sparse.vstack([scipy.sparse.kron(scipy.sparse.eye(len(Eaxis)), scipy.sparse.kron(scipy.sparse.eye(len(muaxis)), LP)), 
                                scipy.sparse.kron(scipy.sparse.eye(len(Eaxis)), scipy.sparse.kron(Lm, scipy.sparse.eye(len(Pphiaxis)))), 
                                scipy.sparse.kron(LE, scipy.sparse.kron(scipy.sparse.eye(len(muaxis)), scipy.sparse.eye(len(Pphiaxis))))]))

    # Preallocate regularisation matrix
    phase_space_dim = len(Eaxis)*len(Pphiaxis)*len(muaxis)
    
    # OLD VERSION!!!
    # kick_strength = kick_matrix[:,-1]

    # for idx in range(kick_matrix.shape[0]):
        
    #     dEidx = int(kick_matrix[idx, 0]) - 1
    #     dPidx = int(kick_matrix[idx, 1]) - 1
    #     Eidx  = int(kick_matrix[idx, 2]) - 1
    #     Pidx  = int(kick_matrix[idx, 3]) - 1
    #     midx  = int(kick_matrix[idx, 4]) - 1
        
    #     dE    = dEaxis[dEidx]
    #     dPphi = dPphiaxis[dPidx] # Should I multiply this with curdir as well? No it is already 'minus dominated'
    #     E     = Eaxis[Eidx]
    #     Pphi  = Pphiaxis[Pidx]
    #     muL   = muaxis[midx]
    #     dmu   = -dE*muL/E # This is actually dLambda
            
    #     Pphi_LFS = P_canonical(np.abs(E), pitch, R_LFS, Z_LFS, mD, Qe, psiwall, Bphi_LFS, B_LFS)
    #     Lambda_LFS = mu(Pphi_LFS, np.abs(E), R_LFS, Z_LFS, mD, Qe, psiwall, Bphi_LFS, B_LFS) * B0/np.abs(E*Qe*1e3)
    #     pphiidx = np.argmin(np.abs(Pphi - Pphi_LFS/Qe/psiwall/curdir))

    #     #Eidx    = np.argmin(np.abs(Eaxis - E))
    #     #Pphiidx = np.argmin(np.abs(Pphiaxis - Pphi))
    #     #muidx   = np.argmin(np.abs(muaxis - mu))

    #     # Check if the orbit is ctr-passing or ctr-stagnation to penalise gradients in sigma=-1 phase space
    #     # if (orbit_types[Eidx,midx,Pidx] == 'ctr-passing') | (orbit_types[Eidx,midx,Pidx] == 'ctr-stagnation'):
    #     if muL < Lambda_LFS[pphiidx]:
    #         Eidx = np.argmin(np.abs(Eaxis + E))

    #     # Convert sub-indices to linear indices
    #     lin_idx = np.ravel_multi_index((Eidx, midx, Pidx), (len(Eaxis), len(muaxis), len(Pphiaxis)))
        
    #     # Make kick vector and projection matrix
    #     kick_vector = np.array([[dPphi/deltaPphi], [dmu/deltamu], [(dE/ENBI)/deltaE]]) # We are actually dealing with lambda direction, so there should be a kick in that direction!!! No kick in the mu-direction
    #     # Since the kicks in each direction are in different units, and we are projecting a finite-difference gradient operator,
    #     # we normalise the kick strengths by the grid spacing in each direction. This should also be done for the normal L1 matrix.
    #     # Otherwise we can for example end up with an orbit receiving a kick in Pphi, but not correlated with its nearest neighbour in Pphi.
    #     # if np.round(np.linalg.norm(kick_vector), 4) > 1e-2:
            
    #     # # Include L1 regularisation outside kick prior
    #     # if np.round(kick_strength[idx], 4) > 0.0:
    #     #     Proj = 10 * kick_strength[idx]/np.max(kick_strength) * kick_vector@(kick_vector.T) / np.linalg.norm(kick_vector) / np.linalg.norm(kick_vector)
    #     # else:
    #     #     Proj = np.eye(3)
    #     # Try not including L1 regularisation outside kick prior
    #     if np.linalg.norm(kick_vector) > 1e-9:
    #         Proj = 1.0 * kick_strength[idx]/np.max(kick_strength) * kick_vector@(kick_vector.T) / np.linalg.norm(kick_vector) / np.linalg.norm(kick_vector)
    #         if Eidx == 2:
    #             print('kick vector = ',kick_vector)
    #     else:
    #         Proj = 0.0 * np.eye(3)
        
    #     # Proj = np.eye(3)
    #     # How do we determine if the flatten from the mode is larger/smaller than the flatting from collisions?
    #     # I mean, what is the corresponding 'kick_strength' associated with the normal L1 regularisation matrix?
    #     # Does it makes sense to say (1+kick_strength[idx]) for the mode? Is this effectively saying the L1 matrix has 'kick strength' of 1?
    #     # Or, what if we just add up the contributions like this?
    #     '''
    #     if np.round(np.linalg.norm(kick_vector), 4) > 1e-2:
    #         Proj = np.eye(3) + kick_strength[idx] * kick_vector@(kick_vector.T) / np.linalg.norm(kick_vector) / np.linalg.norm(kick_vector)
    #     else:
    #         Proj = np.eye(3)
    #     '''
    #     # Project gradient operator onto the kick direction
    #     #temp = Proj@np.concatenate((L1[lin_idx,:], L1[int(phase_space_dim+lin_idx),:], L1[int(2*phase_space_dim+lin_idx),:]), axis=0)
    #     # If sparse matrix
    #     temp = scipy.sparse.csr_matrix(Proj)@scipy.sparse.vstack([L1[[lin_idx],:], L1[[int(phase_space_dim+lin_idx)],:], L1[[int(2*phase_space_dim+lin_idx)],:]])

    #     # Multiply the kick strengths onto each direction and write regularisation matrix
    #     #L[lin_idx,:]                        = temp[0,:]
    #     #L[int(phase_space_dim+lin_idx),:]   = temp[1,:]
    #     #L[int(2*phase_space_dim+lin_idx),:] = temp[2,:]
    #     # If sparse matrix
    #     L1[[lin_idx],:]                        = temp[[0],:]
    #     L1[[int(phase_space_dim+lin_idx)],:]   = temp[[1],:]
    #     L1[[int(2*phase_space_dim+lin_idx)],:] = temp[[2],:]

    # # If sparse matrix
    # return L1
    
    # NEW VERSION!!!
    for idx in range(kick_strength.size):
        if kick_strength[idx] > 0.0:
            Eidx, midx, Pidx = np.unravel_index(idx, shape=shape)
            
            dE = dE_kick_phase_space[idx] * ENBI
            dPphi  = dPphi_kick_phase_space[idx] # Should I multiply this with curdir as well? No it is already 'minus dominated'
            E      = Eaxis[Eidx]
            Pphi   = Pphiaxis[Pidx]
            muL    = muaxis[midx]
            dmu    = -dE*muL/E # This is actually dLambda
            
            # # Ensure penalisation in the white band in between the red and blue kicks
            # if (dE == 0.0) & (dPphi == 0.0):
            #     # Check neighbouring pixels in Pphi direction
            #     if np.abs(dE_kick_phase_space[idx+1] * ENBI) > 0.0:
            #         dE = dE_kick_phase_space[idx+1] * ENBI
            #         dPphi  = dPphi_kick_phase_space[idx+1] # Should I multiply this with curdir as well? No it is already 'minus dominated'
            #         E      = Eaxis[Eidx]
            #         Pphi   = Pphiaxis[Pidx]
            #         muL    = muaxis[midx]
            #         dmu    = -dE*muL/E # This is actually dLambda
            #     elif np.abs(dE_kick_phase_space[idx-1] * ENBI) > 0.0:
            #         dE = dE_kick_phase_space[idx-1] * ENBI
            #         dPphi  = dPphi_kick_phase_space[idx-1] # Should I multiply this with curdir as well? No it is already 'minus dominated'
            #         E      = Eaxis[Eidx]
            #         Pphi   = Pphiaxis[Pidx]
            #         muL    = muaxis[midx]
            #         dmu    = -dE*muL/E # This is actually dLambda
            #     elif np.abs(dE_kick_phase_space[idx+2] * ENBI) > 0.0:
            #         dE = dE_kick_phase_space[idx+2] * ENBI
            #         dPphi  = dPphi_kick_phase_space[idx+2] # Should I multiply this with curdir as well? No it is already 'minus dominated'
            #         E      = Eaxis[Eidx]
            #         Pphi   = Pphiaxis[Pidx]
            #         muL    = muaxis[midx]
            #         dmu    = -dE*muL/E # This is actually dLambda
            #     elif np.abs(dE_kick_phase_space[idx-2] * ENBI) > 0.0:
            #         dE = dE_kick_phase_space[idx-2] * ENBI
            #         dPphi  = dPphi_kick_phase_space[idx-2] # Should I multiply this with curdir as well? No it is already 'minus dominated'
            #         E      = Eaxis[Eidx]
            #         Pphi   = Pphiaxis[Pidx]
            #         muL    = muaxis[midx]
            #         dmu    = -dE*muL/E # This is actually dLambda
            # When dPphi = 0.0, then dE = kick_strength as it should be. I have checked! But then Eslope is inf...
            if dPphi == 0.0:
                # print('\n [Eidx, midx, Pidx] =', [Eidx, midx, Pidx])
                # print('\n kick strength:', kick_strength[idx])
                # print('\n dE = ', dE/ENBI)
                # print('\n Eslope:', Eslope)
                Eslope = dE/ENBI
                Lslope = dmu
                dPphi  = 1.0
            else:
                Eslope = (dE/ENBI)/dPphi
                Lslope = dmu/dPphi
            
            Pphi_LFS = P_canonical(np.abs(E), pitch, R_LFS, Z_LFS, mD, Qe, psiwall, Bphi_LFS, B_LFS)
            Lambda_LFS = mu(Pphi_LFS, np.abs(E), R_LFS, Z_LFS, mD, Qe, psiwall, Bphi_LFS, B_LFS) * B0/np.abs(E*Qe*1e3)
            pphiidx = np.argmin(np.abs(Pphi - Pphi_LFS/Qe/psiwall))
    
            #Eidx    = np.argmin(np.abs(Eaxis - E))
            #Pphiidx = np.argmin(np.abs(Pphiaxis - Pphi))
            #muidx   = np.argmin(np.abs(muaxis - mu))
    
            # Check if the orbit is ctr-passing or ctr-stagnation to penalise gradients in sigma=-1 phase space
            # if (orbit_types[Eidx,midx,Pidx] == 'ctr-passing') | (orbit_types[Eidx,midx,Pidx] == 'ctr-stagnation'):
            if muL < Lambda_LFS[pphiidx]:
                Eidx = np.argmin(np.abs(Eaxis + E))
    
            # Convert sub-indices to linear indices. This is the same as 'idx'. I have checked!
            lin_idx = np.ravel_multi_index((Eidx, midx, Pidx), (len(Eaxis), len(muaxis), len(Pphiaxis)))
            
            # Make kick vector and projection matrix
            kick_vector = np.array([[dPphi/deltaPphi], [dmu/deltamu], [(dE/ENBI)/deltaE]]) # We are actually dealing with lambda direction, so there should be a kick in that direction!!! No kick in the mu-direction
            kick_vector = np.array([[dPphi], [Lslope*dPphi], [Eslope*dPphi]]) # We are actually dealing with lambda direction, so there should be a kick in that direction!!! No kick in the mu-direction
            # Since the kicks in each direction are in different units, and we are projecting a finite-difference gradient operator,
            # we normalise the kick strengths by the grid spacing in each direction. This should also be done for the normal L1 matrix.
            # Otherwise we can for example end up with an orbit receiving a kick in Pphi, but not correlated with its nearest neighbour in Pphi.
                
            # Include L1 regularisation outside kick prior
            if (np.round(kick_strength[idx], 4) > 0.0) | (np.round(kick_strength[idx], 4) <= 0.0):
                if Gazzola == 0:
                    # Use projection method:
                    Proj = 10 * kick_strength[idx]/np.max(kick_strength) * kick_vector@(kick_vector.T) / np.linalg.norm(kick_vector) / np.linalg.norm(kick_vector)
                elif Gazzola == 1:
                    # Try Gazzola method with rotation and scaling June 2024 instead of projection (SEE NOTES IN MY PHYSICAL DTU NOTEBOOK)
                    kick_normed = (kick_vector / np.linalg.norm(kick_vector)).flatten()
                    # Elementary basis vectors
                    e1 = np.array([1.,0.,0.])
                    e2 = np.array([0.,1.,0.])
                    e3 = np.array([0.,0.,1.])
                    # 1. axis is Pphi, 2. axis is Lambda, 3. axis is Energy
                    # Find vector u orthogonal to kick and e3
                    # kick_normed = rotation_matrix(angle=np.pi+np.pi/3, axis=2)@e1 # To match the one in 'make_test_distribution', first rotate by pi, then add (for some reason (is it because this is a right-handed system, and 'make_test_distribution' is left-handed?)) the angle used in 'make_test_distribution'. axis=2 is E here in the global/ravel space. It is axis=0 in unravel space.
                    u = np.cross(kick_normed, e3) / np.linalg.norm(np.cross(kick_normed, e3)) # Sometimes this should have a minus sign, sometimes not... It shouldn't matter, because this is just the axis we rotate about. This results in many row vectors, one for each point in phase space. It seems the minus sign in front works better, even though I thought it should not be there...
                    sinangle = np.linalg.norm(np.cross(kick_normed, e3)) # This is the angle to go from kick_normed to e3. So this provides the angle we need to rotate kick_normed with to get e3.
                    cosangle = np.dot(kick_normed, e3)
                    angle = np.arctan2(sinangle, cosangle)
                    U = np.outer(np.cross(u, e1), e1) + np.outer(np.cross(u, e2), e2) + np.outer(np.cross(u, e3), e3)
                    # Rodgrigues'rotation formula
                    R = np.eye(3) + np.sin(angle)*U + (1-np.cos(angle))*(U@U)
                    # Scaling: (10 times more smoothing along kick than across)
                    spara = - 1/np.log(kick_strength[idx]/np.max(kick_strength) - 1e-6)
                    # S = np.diag([1., 1., 1 + spara])
                    S = np.diag([1., 1., np.sqrt(1 + spara)])
                    # S = np.diag([1., 1., np.sqrt(1+4**2)])
                    # S = np.diag([1., 1., 5.])
                    # Final matrix to be multiplied on L
                    # Proj = kick_strength[idx]/np.max(kick_strength) * S @ R
                    Proj = S @ R
            else:
                Proj = np.eye(3)
            
            # if np.round(kick_strength[idx], 4) > 0.0:
            #     Proj = 10 * kick_strength[idx]/np.max(kick_strength) * kick_vector@(kick_vector.T) / np.linalg.norm(kick_vector) / np.linalg.norm(kick_vector)
            # else:
            #     Proj = np.eye(3)
            
            # Proj = np.eye(3)
            # How do we determine if the flatten from the mode is larger/smaller than the flatting from collisions?
            # I mean, what is the corresponding 'kick_strength' associated with the normal L1 regularisation matrix?
            # Does it makes sense to say (1+kick_strength[idx]) for the mode? Is this effectively saying the L1 matrix has 'kick strength' of 1?
            # Or, what if we just add up the contributions like this?
            '''
            if np.round(np.linalg.norm(kick_vector), 4) > 1e-2:
                Proj = np.eye(3) + kick_strength[idx] * kick_vector@(kick_vector.T) / np.linalg.norm(kick_vector) / np.linalg.norm(kick_vector)
            else:
                Proj = np.eye(3)
            '''
            # Project gradient operator onto the kick direction
            #temp = Proj@np.concatenate((L1[lin_idx,:], L1[int(phase_space_dim+lin_idx),:], L1[int(2*phase_space_dim+lin_idx),:]), axis=0)
            # If sparse matrix
            temp = scipy.sparse.csr_matrix(Proj)@scipy.sparse.vstack([L1[[lin_idx],:], L1[[int(phase_space_dim+lin_idx)],:], L1[[int(2*phase_space_dim+lin_idx)],:]])
    
            # Multiply the kick strengths onto each direction and write regularisation matrix
            #L[lin_idx,:]                        = temp[0,:]
            #L[int(phase_space_dim+lin_idx),:]   = temp[1,:]
            #L[int(2*phase_space_dim+lin_idx),:] = temp[2,:]
            # If sparse matrix
            L1[[lin_idx],:]                        = temp[[0],:]
            L1[[int(phase_space_dim+lin_idx)],:]   = temp[[1],:]
            L1[[int(2*phase_space_dim+lin_idx)],:] = temp[[2],:]

    return L1

# =============================================================================

def plot_precision_matrix(L, Eaxis, muaxis, Pphiaxis):

    from scipy.constants import physical_constants as const
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt

    q = const['elementary charge'][0]
    curdir = -1

    # prec = ((L.T)@L).todense()
    prec = ((L.T)@L)

    fig = plt.figure()
    ax  = fig.add_subplot(111)    # The big subplot
    # Number of subplots: 4 rows for 4 triplets. 4 columns for 4 energy slices for each triplet (an orbit can be correlated with orbits with different energy)
    ax1 = fig.add_subplot(3, 3, 1)
    ax2 = fig.add_subplot(3, 3, 2)
    ax3 = fig.add_subplot(3, 3, 3)
    ax4 = fig.add_subplot(3, 3, 4)
    ax5 = fig.add_subplot(3, 3, 5)
    ax6 = fig.add_subplot(3, 3, 6)
    ax7 = fig.add_subplot(3, 3, 7)
    ax8 = fig.add_subplot(3, 3, 8)
    ax9 = fig.add_subplot(3, 3, 9)
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

    # Triplets to plot (choose 3)
    E = [2.6, 10, 19]
    Pphi = [-1.2/curdir, -1.3/curdir, -1.5/curdir]
    Lambda = [0.25, 0.25, 0.25]
    # Find indices
    Eidxs = np.argmin(np.abs(np.array(Eaxis)[:,None] - E), axis=0)
    Pidxs = np.argmin(np.abs(np.array(Pphiaxis)[:,None] - Pphi), axis=0)
    Lidxs = np.argmin(np.abs(np.array(muaxis)[:,None] - Lambda), axis=0)
    lin_idxs = np.ravel_multi_index((Eidxs, Lidxs, Pidxs), (len(Eaxis), len(muaxis), len(Pphiaxis)))
    # Plot
    ax_idx = 0
    for i, idx in enumerate(lin_idxs):
        # Reshape to 3D
        prec1D = prec[:,[idx]].todense()
        print('shape:', prec1D.shape)
        prec2D = prec1D.reshape((len(Eaxis), len(muaxis) * len(Pphiaxis)))
        for e in range(3):
            im = axs[ax_idx].imshow(np.fliplr(prec2D[int(Eidxs[i]-1+e),:].reshape((len(muaxis),len(Pphiaxis)))), cmap='seismic', extent=(Pphiaxis[-1]/curdir,Pphiaxis[0]/curdir,muaxis[0],muaxis[-1]), origin='lower', vmin=-np.max(np.abs(prec2D)), vmax=np.max(np.abs(prec2D)))
            divider = make_axes_locatable(axs[ax_idx])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

            ax_idx += 1

    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_ylabel('Norm. magnetic moment, $\\Lambda=\\mu B_{0}/E$', fontsize=14)
    ax.set_xlabel('Norm. tor. can. angular momentum, $P_{\\phi}/q|\\Psi_{w}|$', fontsize=14)
    plt.ion()
    plt.show()

# ========================================================================================

# Function to plot a plane of constant energy of the kick probabilities

# Inputs needed:
#   - Kick matrix as returned by the read_kick_matrix(filename) function.
#   - Energy value in units of keV of the plane to plot

# Output:
#   - Plots the kick probabilities in energy an Pphi in a plane of constant energy given as input

def plot_kicks(kick_dict, ENBI):

    # This functions should probably be in the tomo scripts folder, where the equilibrium is also loaded and such
    from scipy.constants import physical_constants as const
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py

    q = const['elementary charge'][0]
    
    # Unpack various variables dict
    various_variables_dict = h5py.File('C:/Users/larsen/Documents/Analytical orbit tomography/Resonant mode-particle interactions/Tomography/FIDA/86327/Time_0-48/various_variables_dict_86327_0.480.h5', 'r')
    Pphi_axis = various_variables_dict['Pphi_axis'][()]
    Pphi_trap = various_variables_dict['Pphi_trap'][()]
    Qe = various_variables_dict['q'][()]
    mD = various_variables_dict['m'][()]
    psi = various_variables_dict['psi'][()]
    psiwall = various_variables_dict['psiwall'][()]
    psiwall_norm = various_variables_dict['psiwall_norm'][()]
    psi0 = various_variables_dict['psi0'][()]/2/np.pi
    B = various_variables_dict['Babs'][()]
    B0 = various_variables_dict['B0'][()]
    B_LFS = various_variables_dict['B_LFS'][()]
    B_HFS = various_variables_dict['B_HFS'][()]
    Bphi = various_variables_dict['Bphi'][()]
    Bphi0 = various_variables_dict['Bphi0'][()]
    Bphi_LFS = various_variables_dict['Bphi_LFS'][()]
    Bphi_HFS = various_variables_dict['Bphi_HFS'][()]
    B_trap = various_variables_dict['B_trap'][()]
    R = various_variables_dict['R'][()]
    R_LFS = various_variables_dict['R_LFS'][()]
    R_HFS = various_variables_dict['R_HFS'][()]
    R0 = various_variables_dict['R0'][()]
    Z_LFS = various_variables_dict['Z_LFS'][()]
    Z_HFS = various_variables_dict['Z_HFS'][()]
    Rvec = various_variables_dict['Rvec'][()]
    zvec = various_variables_dict['zvec'][()]
    axisR = various_variables_dict['axisR'][()]
    axisZ = various_variables_dict['axisZ'][()]
    pitch = various_variables_dict['pitch'][()]
    various_variables_dict.close()
    curdir = -1
    plasma_mask = (Bphi > Bphi_LFS) & (Bphi < Bphi_HFS) & (psi < psiwall) # Inside plasma volume
    outside_mask = (Bphi < Bphi_LFS) | (Bphi > Bphi_HFS) | (psi > psiwall) # Outside plasma volume
    R, Z = np.meshgrid(Rvec, zvec)

    # Unpack dictionary
    kick_phase_space       = kick_dict['kick_phase_space'] # 3D array, with the kick strength in each point in (E,Pphi,mu) phase space
    dE_kick_phase_space    = kick_dict['dE_kick_phase_space']
    dPphi_kick_phase_space = kick_dict['dPphi_kick_phase_space']
    dEaxis                 = kick_dict['dEaxis']
    dPphiaxis              = kick_dict['dPphiaxis']
    Eaxis                  = kick_dict['Eaxis'] / ENBI # This is in keV
    Pphiaxis               = kick_dict['Pphiaxis'] # This is normalised toroidal canonical angular momentum
    muaxis                 = kick_dict['muaxis']   # This is normalised magnetic moment
    
    # Check if using Marios default kick matrix
    if (Pphiaxis.size == 39) & (Eaxis.size == 15) & (muaxis.size == 15):
        kick_phase_space = np.flip(kick_phase_space, axis=2)
        dE_kick_phase_space = np.flip(dE_kick_phase_space, axis=2)
        dPphi_kick_phase_space = np.flip(dPphi_kick_phase_space, axis=2)
        Pphiaxis = -np.flip(Pphiaxis)
    
    PPHI, MU = np.meshgrid(Pphiaxis, muaxis)
    # Fine Pphi axis
    Pphi_fine = np.linspace(Pphiaxis[0], Pphiaxis[-1], 200)
    
    # Grid spacing
    deltaE = np.mean(np.diff(Eaxis[len(Eaxis)//2:])) # Normalise with NBI energy such that all dimensions are of order 1.
    deltaPphi = np.mean(np.diff(Pphiaxis))
    deltamu = np.mean(np.diff(muaxis))
    
    # For saving
    foldername = "C:/Users/larsen/Documents/Analytical orbit tomography/Resonant mode-particle interactions/Kick matrices/86327/Time_1-05/"
    
    for eidx, E in enumerate(Eaxis[::2]):
        if E > 0.0:
            e = np.argmin(np.abs(np.array(Eaxis) - E))
            
            Lambda_trap = (np.abs(E*ENBI)*1e3*Qe/B_trap)*B0/np.abs(E*ENBI*Qe*1e3)
            
            Pphi_LFS = P_canonical(np.abs(E*ENBI), pitch, R_LFS, Z_LFS, mD, Qe, psiwall, Bphi_LFS, B_LFS)
            Lambda_LFS = mu(Pphi_LFS, np.abs(E*ENBI), R_LFS, Z_LFS, mD, Qe, psiwall, Bphi_LFS, B_LFS) * B0/np.abs(E*ENBI*Qe*1e3)
            Pphi_HFS = P_canonical(np.abs(E*ENBI), pitch, R_HFS, Z_HFS, mD, Qe, psiwall, Bphi_HFS, B_HFS)
            Lambda_HFS = mu(Pphi_HFS, np.abs(E*ENBI), R_HFS, Z_HFS, mD, Qe, psiwall, Bphi_HFS, B_HFS) * B0/np.abs(E*ENBI*Qe*1e3)
            Pphi0 = P_canonical(np.abs(E*ENBI), pitch, R0, 0, mD, Qe, 0, Bphi0, B0)
            Lambda0 = mu(Pphi0, np.abs(E*ENBI), R0, 0, mD, Qe, 0, Bphi0, B0) * B0/np.abs(E*ENBI*Qe*1e3)
            # Maximum Lambda
            Lambda_max = np.zeros(Pphi_fine.size)
            for i, p in enumerate(Pphi_fine): 
                Lambda_max[i] = np.max(mu(p*psiwall*Qe, np.abs(E*ENBI), R[plasma_mask], Z[plasma_mask], mD, Qe, psi[plasma_mask], Bphi[plasma_mask], B[plasma_mask]) * B0/np.abs(E*ENBI*Qe*1e3))
            
            # Kick quantities to plot
            Pphi_thing_to_plot = np.flip(dPphi_kick_phase_space[e,:,:]/curdir, axis=-1)
            E_thing_to_plot = np.flip(dE_kick_phase_space[e,:,:], axis=-1)
            mask = Pphi_thing_to_plot != E_thing_to_plot
            kick_to_plot = np.flip(kick_phase_space[e,:,:], axis=-1) / np.max(kick_phase_space)
            # other_mask = Pphi_thing_to_plot == E_thing_to_plot
            # kick_to_plot[other_mask] = np.nan # The plot becomes ugly with this
            
            # Plot strength of kicks
            fig, ax = plt.subplots()
            # Plot parabolas and trapped boundary
            plt.plot(Pphi_fine/curdir, Lambda_max, 'k-')
            plt.plot(Pphi_LFS/Qe/(psiwall*curdir),Lambda_LFS,'k-')
            plt.plot(Pphi_HFS/Qe/(psiwall*curdir),Lambda_HFS,'k--')
            plt.plot(Pphi_trap/Qe/(psiwall*curdir),Lambda_trap,'k-')
            plt.plot(Pphi0/Qe/(psiwall*curdir),Lambda0,'k-.')
            plt.plot(-np.ones(10),np.linspace(np.min(Lambda_trap),np.max(Lambda_trap),10),'k-')
            # Plot kicks
            im = plt.imshow(kick_to_plot, cmap='gist_stern_r', extent=((Pphiaxis[-1]/curdir,Pphiaxis[0]/curdir,muaxis[0],muaxis[-1])), origin='lower', vmin=0, vmax=1)#, interpolation='hermite')
            ax.set_xlim([np.min(Pphi_fine/curdir), np.max(Pphi_fine/curdir)])
            ax.set_ylim([muaxis[0], muaxis[-1]])
            ax.set_xlabel('Norm. tor. can. angular momentum, $P_{\\phi}/q|\\Psi_{w}|$', fontsize=14)
            ax.set_ylabel('$\\Lambda=\\mu B_{0}/E$', fontsize=14)
            # ax.set_title(f'Kick strength, energy, E={np.round(E*ENBI,2)} keV', fontsize=14)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical', label='Kick mag. [a.u.]')
            cbar.ax.yaxis.label.set_size(14)
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            plt.subplots_adjust(left=0.045, bottom=0.06, right=0.941, top=0.969, wspace=0.2, hspace=0.2)
            # For saving
            filename = f'Kick_strength_E{(E*ENBI):.2f}keV'
            # plt.savefig(foldername+filename+'.png', bbox_inches='tight')
            # plt.savefig(foldername+filename+'.eps', bbox_inches='tight')
            # plt.savefig(foldername+filename+'.svg', bbox_inches='tight')
    
            # Plot dE
            fig, ax = plt.subplots()
            # Plot parabolas and trapped boundary
            plt.plot(Pphi_fine/curdir, Lambda_max, 'k-')
            plt.plot(Pphi_LFS/Qe/(psiwall*curdir),Lambda_LFS,'k-')
            plt.plot(Pphi_HFS/Qe/(psiwall*curdir),Lambda_HFS,'k--')
            plt.plot(Pphi_trap/Qe/(psiwall*curdir),Lambda_trap,'k-')
            plt.plot(Pphi0/Qe/(psiwall*curdir),Lambda0,'k-.')
            plt.plot(-np.ones(10),np.linspace(np.min(Lambda_trap),np.max(Lambda_trap),10),'k-')
            # Plot kicks
            im = plt.imshow(np.flip(dE_kick_phase_space[e,:,:], axis=-1)*ENBI, cmap='seismic', extent=((Pphiaxis[-1]/curdir,Pphiaxis[0]/curdir,muaxis[0],muaxis[-1])), origin='lower', vmin=-np.max(np.abs(dE_kick_phase_space))*ENBI, vmax=np.max(np.abs(dE_kick_phase_space))*ENBI)#, interpolation='hermite')
            # Plot vector arrows
            plt.quiver(np.fliplr(PPHI/curdir)[mask], MU[mask], Pphi_thing_to_plot[mask], np.fliplr(-(MU/E))[mask] * E_thing_to_plot[mask], scale=3.0)
            ax.set_xlim([np.min(Pphi_fine/curdir), np.max(Pphi_fine/curdir)])
            ax.set_ylim([muaxis[0], muaxis[-1]])
            ax.set_xlabel('Norm. tor. can. angular momentum, $P_{\\phi}/q|\\Psi_{w}|$', fontsize=14)
            ax.set_ylabel('$\\Lambda=\\mu B_{0}/E$', fontsize=14)
            # ax.set_title(f'$dE$ kick, energy, E={np.round(E*ENBI,2)} keV', fontsize=14)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical', label='$dE$ [keV]') # I have multiplied with ENBI in the plot just so the colorbar has units of keV instead of normalised units.
            cbar.ax.yaxis.label.set_size(14)
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            plt.subplots_adjust(left=0.045, bottom=0.06, right=0.941, top=0.969, wspace=0.2, hspace=0.2)
            # For saving
            filename = f'dE_kick_E{(E*ENBI):.2f}keV'
            # plt.savefig(foldername+filename+'.png', bbox_inches='tight')
            # plt.savefig(foldername+filename+'.eps', bbox_inches='tight')
            # plt.savefig(foldername+filename+'.svg', bbox_inches='tight')
    
            # Plot dPphi
            fig, ax = plt.subplots()
            # Plot parabolas and trapped boundary
            plt.plot(Pphi_fine/curdir, Lambda_max, 'k-')
            plt.plot(Pphi_LFS/Qe/(psiwall*curdir),Lambda_LFS,'k-')
            plt.plot(Pphi_HFS/Qe/(psiwall*curdir),Lambda_HFS,'k--')
            plt.plot(Pphi_trap/Qe/(psiwall*curdir),Lambda_trap,'k-')
            plt.plot(Pphi0/Qe/(psiwall*curdir),Lambda0,'k-.')
            plt.plot(-np.ones(10),np.linspace(np.min(Lambda_trap),np.max(Lambda_trap),10),'k-')
            # Plot kicks
            im = plt.imshow(Pphi_thing_to_plot, cmap='seismic', extent=((Pphiaxis[-1]/curdir,Pphiaxis[0]/curdir,muaxis[0],muaxis[-1])), origin='lower', vmin=-np.max(np.abs(dPphi_kick_phase_space)), vmax=np.max(np.abs(dPphi_kick_phase_space)))#, interpolation='hermite')
            # Plot vector arrows
            plt.quiver(np.fliplr(PPHI/curdir)[mask], MU[mask], Pphi_thing_to_plot[mask], np.fliplr(-(MU/E))[mask] * E_thing_to_plot[mask], scale=3.0)
            ax.set_xlim([np.min(Pphi_fine/curdir), np.max(Pphi_fine/curdir)])
            ax.set_ylim([muaxis[0], muaxis[-1]])
            ax.set_xlabel('Norm. tor. can. angular momentum, $P_{\\phi}/q|\\Psi_{w}|$', fontsize=14)
            ax.set_ylabel('$\\Lambda=\\mu B_{0}/E$', fontsize=14)
            # ax.set_title('$dP_{\\phi}$ kick, energy, '+f'E={np.round(E*ENBI,2)} keV', fontsize=14)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical', label='$dP_{\\phi}/q\\Psi_{w}$ [-]')
            cbar.ax.yaxis.label.set_size(14)
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            plt.subplots_adjust(left=0.045, bottom=0.06, right=0.941, top=0.969, wspace=0.2, hspace=0.2)
            # For saving
            filename = f'dPphi_kick_E{(E*ENBI):.2f}keV'
            # plt.savefig(foldername+filename+'.png', bbox_inches='tight')
            # plt.savefig(foldername+filename+'.eps', bbox_inches='tight')
            # plt.savefig(foldername+filename+'.svg', bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    
    ENBI = 28.0

    kick_dict   = read_kick_matrix('Kick matrices/pDEDP_a5p0.AEP', ENBI)

    L = make_kick_regularisation_matrix(kick_dict, ENBI)

    # Unpack dictionary
    kick_matrix = kick_dict['kick_prob_array']
    dEaxis      = kick_dict['dEaxis']
    dPphiaxis   = kick_dict['dPphiaxis']
    Eaxis       = kick_dict['Eaxis']
    Pphiaxis    = kick_dict['Pphiaxis']
    muaxis      = kick_dict['muaxis']
    
    # Plot a plane of constant energy of the kick matrix
    plot_kicks(kick_dict, ENBI)