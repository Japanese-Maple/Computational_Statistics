import numpy as np
from scipy import sparse
from scipy.sparse import bmat, linalg

import numpy as np
from scipy import sparse

#=================================================================================================================
# Main Mesh Processing Functions
#=================================================================================================================

def refine(p, e, t): # by Stefan Takacs
    """
    Uniformly refine mesh by subdividing all triangles into 4 congruent ones.
    """
    Np = p.shape[0]
    Ne = e.shape[0]
    Nt = t.shape[0]
    
    # All new data structures preserve the number of colums;
    # columns of e and t without special meaning are inherited
    # by children.
    pnew = np.zeros((Np+Ne,    p.shape[1]))
    enew = np.zeros((2*Ne+3*Nt,e.shape[1]),'i')
    tnew = np.zeros((4*Nt,     t.shape[1]),'i')
    
    # New points are
    #  a) old points (with old indices) 
    pnew[:Np,:] = p
    #  b) new points on midpoint of edges Ei (with index Np+Ei)
    pnew[Np:Np+Ne,:] = (p[e[:,0],:] + p[e[:,1],:])/2

    # New edges are
    #  a) in place of old edge Ei between P0 and P1
    #    aa) new edge 2*i   between P0 and midpoint of Ei
    enew[0:2*Ne:2,:] = e[:,:]
    enew[0:2*Ne:2,1] = range(Np,Np+Ne)
    #    ab) new edge 2*i+1 between P1 and midpoint of Ei
    enew[1:2*Ne:2,:] = e[:,:]
    enew[1:2*Ne:2,0] = range(Np,Np+Ne)
    #  b) inside of triangle i with edges E0, E1 and E2
    #    ba) new edge 2*Ne+3*i   between midpoint of E0 and midpoint of E1
    enew[2*Ne  :2*Ne+3*Nt:3,0] = Np+t[:,3+0]
    enew[2*Ne  :2*Ne+3*Nt:3,1] = Np+t[:,3+1]
    #    bb) new edge 2*Ne+3*i+1 between midpoint of E1 and midpoint of E2
    enew[2*Ne+1:2*Ne+3*Nt:3,0] = Np+t[:,3+1]
    enew[2*Ne+1:2*Ne+3*Nt:3,1] = Np+t[:,3+2]
    #    bc) new edge 2*Ne+3*i+2 between midpoint of E2 and midpoint of E0
    enew[2*Ne+2:2*Ne+3*Nt:3,0] = Np+t[:,3+2]
    enew[2*Ne+2:2*Ne+3*Nt:3,1] = Np+t[:,3+0]
    
    # New triangles are in place of old triangle Ti with
    #   a) corners P0, midpoint of E0 and midpoint of E2
    tnew[0:4*Nt:4,:] = t[:,:]
    tnew[0:4*Nt:4,0] = t[:,0]
    tnew[0:4*Nt:4,1] = Np+t[:,3+0]
    tnew[0:4*Nt:4,2] = Np+t[:,3+2]
    tnew[0:4*Nt:4,3] = 2*t[:,3+0] + ( enew[2*t[:,3+0]+1,1] == t[:,0] )
    tnew[0:4*Nt:4,4] = 2*Ne+3*np.arange(Nt)+2
    tnew[0:4*Nt:4,5] = 2*t[:,3+2] + ( enew[2*t[:,3+2]+1,1] == t[:,0] )
    #   b) corners P1, midpoint of E1 and midpoint of E0
    tnew[1:4*Nt:4,:] = t[:,:]
    tnew[1:4*Nt:4,0] = t[:,1]
    tnew[1:4*Nt:4,1] = Np+t[:,3+1]
    tnew[1:4*Nt:4,2] = Np+t[:,3+0]
    tnew[1:4*Nt:4,3] = 2*t[:,3+1] + ( enew[2*t[:,3+1]+1,1] == t[:,1] )
    tnew[1:4*Nt:4,4] = 2*Ne+3*np.arange(Nt)+0
    tnew[1:4*Nt:4,5] = 2*t[:,3+0] + ( enew[2*t[:,3+0]+1,1] == t[:,1] )
    #   c) corners P2, midpoint of E2 and midpoint of E1
    tnew[2:4*Nt:4,:] = t[:,:]
    tnew[2:4*Nt:4,0] = t[:,2]
    tnew[2:4*Nt:4,1] = Np+t[:,3+2]
    tnew[2:4*Nt:4,2] = Np+t[:,3+1]
    tnew[2:4*Nt:4,3] = 2*t[:,3+2] + ( enew[2*t[:,3+2]+1,1] == t[:,2] )
    tnew[2:4*Nt:4,4] = 2*Ne+3*np.arange(Nt)+1
    tnew[2:4*Nt:4,5] = 2*t[:,3+1] + ( enew[2*t[:,3+1]+1,1] == t[:,2] )
    #   d) corners midpoint of E0, midpoint of E1 and midpoint of E2
    tnew[3:4*Nt:4,:] = t[:,:]
    tnew[3:4*Nt:4,0] = Np+t[:,3+0]
    tnew[3:4*Nt:4,1] = Np+t[:,3+1]
    tnew[3:4*Nt:4,2] = Np+t[:,3+2]
    tnew[3:4*Nt:4,3] = 2*Ne+3*np.arange(Nt)+0 ### TODO: set to int or use range
    tnew[3:4*Nt:4,4] = 2*Ne+3*np.arange(Nt)+1
    tnew[3:4*Nt:4,5] = 2*Ne+3*np.arange(Nt)+2
    
    return [pnew, enew, tnew]

def refine_n_times(p, e, t, number_of_refinements: int= 3):
    """
    refines the mesh n times
    """
    for i in range(number_of_refinements):
        p, e, t = refine(p, e, t)
        
    return p, e, t

def fix_orientation(p, tri):
    tri = tri.copy()

    det = (p[tri[:, 1], 0] - p[tri[:, 0], 0]) * (p[tri[:, 2], 1] - p[tri[:, 0], 1]) - \
          (p[tri[:, 2], 0] - p[tri[:, 0], 0]) * (p[tri[:, 1], 1] - p[tri[:, 0], 1])

    flip = det < 0

    tmp = tri[flip, 1].copy()
    tri[flip, 1] = tri[flip, 2]
    tri[flip, 2] = tmp

    return tri

def build_stable_mesh(p, tri_idx):
    tri_idx = fix_orientation(p, tri_idx)

    edge_dict = {}
    edges = []
    tri_edges = []

    for tri in tri_idx:
        local_edges = []

        for i in range(3):
            a = tri[i]
            b = tri[(i + 1) % 3]
            edge = tuple(sorted((a, b)))

            if edge not in edge_dict:
                edge_dict[edge] = len(edges)
                edges.append(edge)

            local_edges.append(edge_dict[edge])

        tri_edges.append(local_edges)

    e = np.zeros((len(edges), 3), dtype=int)
    e[:, :2] = np.array(edges)

    t = np.zeros((tri_idx.shape[0], 7), dtype=int)
    t[:, :3] = tri_idx
    t[:, 3:6] = np.array(tri_edges)

    # Mark boundary edges
    counts = np.zeros(len(edges), dtype=int)
    for tri in t:
        for i in range(3):
            counts[tri[3+i]] += 1

    e[counts == 1, 2] = 1  # boundary flag

    return p, e, t

#===============================================================================================================================================================
# MAIN COMPUTATIONAL FUNCTIONS
#===============================================================================================================================================================

def calculate_velocity_A(p, t, kinematic_viscosity):
    """Calculates the Stiffness Matrix **A**"""

    Np = p.shape[0]
    Nt = t.shape[0]

    # we calculate all jacobians simultaneously           J = |x_2 - x_1   x_3 - x_1|
    #                                                         |y_2 - y_1   y_3 - y_1|

    jacobian = np.zeros(shape=(Nt, 2, 2))
    jacobian[:, 0, :] = p[t[:, 1]] - p[t[:, 0]] 
    jacobian[:, 1, :] = p[t[:, 2]] - p[t[:, 0]] 

    det_J = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]

    # Cofactor multiplication matrix Q:                   Q = |(y_3 - y_1)^2 + (x_3 - x_1)^2                   (y_1 - y_2)(y_3 - y_1) + (x_2 - x_1)(x_1 - x_3)|
    #                                                         |(y_1 - y_2)(y_3 - y_1) + (x_2 - x_1)(x_1 - x_3)  (y_2 - y_1)^2 + (x_2 - x_1)^2                 |
    #
    #                                                     Q = Cof(J).T @ Cof(J)

    q11 = jacobian[:, 1, 1]**2 + jacobian[:, 1, 0]**2
    q12 = -(jacobian[:, 1, 1]*jacobian[:, 0, 1] + jacobian[:, 1, 0]*jacobian[:, 0, 0])
    q22 = jacobian[:, 0, 1]**2 + jacobian[:, 0, 0]**2

    Q_mat = np.zeros_like(jacobian)
    Q_mat[:, 0, 0] = q11
    Q_mat[:, 1, 0], Q_mat[:, 0, 1] = q12, q12
    Q_mat[:, 1, 1] = q22
    
    test_function_derivatives = np.array([[-1, -1],   # ф1 = 1 - s_1 - s_2
                                          [ 1,  0],   # ф2 = s_1
                                          [ 0,  1]])  # ф3 = s_2


    # We can now construct a local matrix A for each triangle:
    
    A_local = np.einsum('mi,tij,nj->tmn', test_function_derivatives, Q_mat, test_function_derivatives)
    A_local *= (kinematic_viscosity / (2.0 * det_J))[:, None, None]

    rowidx = np.einsum("ni,j->nij", t[:,0:3], [1,1,1])
    colidx = np.einsum("nj,i->nij", t[:,0:3], [1,1,1])
    
    # Return corresponding csc_matrix
    return sparse.csc_matrix((np.ravel(A_local),(np.ravel(rowidx),np.ravel(colidx))),shape=(Np,Np))

# def calculate_velocity_A(p, t, kinematic_viscosity):
#     """Calculates the Stiffness Matrix **A**"""
#     Np = p.shape[0]
#     Nt = t.shape[0]

#     jacobian = np.zeros(shape=(Nt, 2, 2))
#     jacobian[:, 0, :] = p[t[:, 1]] - p[t[:, 0]]   # (x2-x1, y2-y1)
#     jacobian[:, 1, :] = p[t[:, 2]] - p[t[:, 0]]   # (x3-x1, y3-y1)

#     det_J = jacobian[:, 0, 0]*jacobian[:, 1, 1] - jacobian[:, 0, 1]*jacobian[:, 1, 0]

#     q11 = jacobian[:, 1, 1]**2 + jacobian[:, 1, 0]**2
#     q12 = -(jacobian[:, 1, 1]*jacobian[:, 0, 1] + jacobian[:, 1, 0]*jacobian[:, 0, 0])
#     q22 = jacobian[:, 0, 1]**2 + jacobian[:, 0, 0]**2

#     Q_mat = np.zeros_like(jacobian)
#     Q_mat[:, 0, 0] = q11
#     Q_mat[:, 1, 0], Q_mat[:, 0, 1] = q12, q12
#     Q_mat[:, 1, 1] = q22

#     test_function_derivatives = np.array([[-1., -1.],
#                                           [ 1.,  0.],
#                                           [ 0.,  1.]])

#     A_local = np.einsum('mi,tij,nj->tmn',
#                         test_function_derivatives, Q_mat, test_function_derivatives)
#     A_local *= (kinematic_viscosity / (2.0*np.abs(det_J)))[:, None, None]

#     rowidx = np.einsum("ni,j->nij", t[:, 0:3], [1, 1, 1])
#     colidx = np.einsum("nj,i->nij", t[:, 0:3], [1, 1, 1])
#     return sparse.csc_matrix((np.ravel(A_local), (np.ravel(rowidx), np.ravel(colidx))),
#                              shape=(Np, Np))

#_______________________________________________________________________________________________________________________________________________________________

def calculate_mass_M(p, t,):

    Np = p.shape[0]
    Nt = t.shape[0]

    jacobian = np.zeros(shape=(Nt, 2, 2))
    jacobian[:, 0, :] = p[t[:, 1]] - p[t[:, 0]]
    jacobian[:, 1, :] = p[t[:, 2]] - p[t[:, 0]]

    det_J = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]

    M_local = np.einsum("n,ij->nij", 
                        np.abs(det_J)/24, 
                        np.array([
                            [2, 1, 1],
                            [1, 2, 1],
                            [1, 1, 2]
                        ]))

    rowidx = np.einsum("ni,j->nij", t[:,0:3], [1,1,1])
    colidx = np.einsum("nj,i->nij", t[:,0:3], [1,1,1])

    return sparse.csc_matrix((np.ravel(M_local),(np.ravel(rowidx),np.ravel(colidx))),shape=(Np,Np))

#_______________________________________________________________________________________________________________________________________________________________


# def calculate_pressure_B(p_fine, t_fine, p_coarse, t_coarse):
#     """Calculates the pressure matrices **Bx**, **By**"""

#     Np_fine = p_fine.shape[0]
#     Nt_fine = t_fine.shape[0]

#     Np_coarse = p_coarse.shape[0]
#     Nt_coarse = t_coarse.shape[0]

#     jacobian = np.zeros(shape=(Nt_fine, 2, 2))
#     jacobian[:, 0, :] = p_fine[t_fine[:, 1]] - p_fine[t_fine[:, 0]] 
#     jacobian[:, 1, :] = p_fine[t_fine[:, 2]] - p_fine[t_fine[:, 0]] 

#     test_function_derivatives = np.array([[-1, -1],   # ф1 = 1 - s_1 - s_2
#                                           [ 1,  0],   # ф2 = s_1
#                                           [ 0,  1]])  # ф3 = s_2
    
#     # We can now construct local matrices Bx, By for each triangle:

#     Bx_local = np.zeros(shape=(Nt_fine, 3, 3))
#     By_local = np.zeros(shape=(Nt_fine, 3, 3))
            
#     Bx_vals = 1/6 * (  jacobian[:, 0, 1, None] * test_function_derivatives[:, 1]       
#                      - jacobian[:, 1, 1, None] * test_function_derivatives[:, 0])
    
#     By_vals = 1/6 * (  jacobian[:, 1, 0, None] * test_function_derivatives[:, 0] 
#                      - jacobian[:, 0, 0, None] * test_function_derivatives[:, 1])
    
#     Bx_local = np.repeat(Bx_vals[:, :, None], 3, axis=2)
#     By_local = np.repeat(By_vals[:, :, None], 3, axis=2)

#     # We now adress the global matrix problem for Bx and By:

#     fine_to_coarse_idx = np.repeat(np.arange(Nt_coarse), 4)[:Nt_fine]
#     colidx = np.tile(t_fine[:, :3, None], (1, 1, 3)).ravel()
#     parent_coarse_nodes = t_coarse[fine_to_coarse_idx, :3]
#     rowidx = np.tile(parent_coarse_nodes[:, None, :], (1, 3, 1)).ravel()
    
#     B_x = sparse.csc_matrix((np.ravel(Bx_local),
#                             (np.ravel(rowidx), np.ravel(colidx))),
#                             shape=(Np_coarse, Np_fine))
    
#     B_y = sparse.csc_matrix((np.ravel(By_local),
#                             (np.ravel(rowidx), np.ravel(colidx))),
#                             shape=(Np_coarse, Np_fine))

#     return B_x, B_y


# def calculate_pressure_B(p_fine, t_fine, p_coarse, t_coarse):

#     Nt_fine   = t_fine.shape[0]
#     Np_fine   = p_fine.shape[0]

#     Nt_coarse = t_coarse.shape[0]
#     Np_coarse = p_coarse.shape[0]

#     fine_to_coarse = np.repeat(np.arange(Nt_coarse), 4)[:Nt_fine]

#     # Calculating fine-triangle & coarse-triangle nodes
#     tf = t_fine[:, :3]
#     tc = t_coarse[fine_to_coarse, :3]

#     pf = p_fine[tf]

#     # Calculating Jacobians
#     Jf = np.stack([pf[:, 1] - pf[:, 0],
#                    pf[:, 2] - pf[:, 0]], axis=2)
  
#     det_Jf = Jf[:, 0, 0] * Jf[:, 1, 1] - Jf[:, 0, 1] * Jf[:, 1, 0]
#     area  = 0.5 * np.abs(det_Jf)

#     # Gradients via closed-form 2×2 inverse
#     inv_det_Jf = 1.0 / det_Jf                                            

#     inv_Jf_T = np.empty((Nt_fine, 2, 2))
#     inv_Jf_T[:, 0, 0] =  Jf[:, 1, 1] * inv_det_Jf
#     inv_Jf_T[:, 0, 1] = -Jf[:, 1, 0] * inv_det_Jf
#     inv_Jf_T[:, 1, 0] = -Jf[:, 0, 1] * inv_det_Jf
#     inv_Jf_T[:, 1, 1] =  Jf[:, 0, 0] * inv_det_Jf

#     test_function_derivatives = np.array([[-1, -1],
#                                           [ 1,  0],
#                                           [ 0,  1]])

#     grads = np.einsum('ndk,ik->nid', inv_Jf_T, test_function_derivatives)

#     # Precomputed constant values of psi at the 4 child centroids
#     psi_children = np.array([
#         [2/3, 1/6, 1/6],  # T1
#         [1/6, 2/3, 1/6],  # T2
#         [1/6, 1/6, 2/3],  # T3
#         [1/3, 1/3, 1/3]   # T4
#     ])

#     psi = np.tile(psi_children, (Nt_coarse, 1))[:Nt_fine] 

#     # Local B blocks
#     Bx_loc = -area[:, None, None] * np.einsum('ni,nj->nij', psi, grads[:, :, 0])
#     By_loc = -area[:, None, None] * np.einsum('ni,nj->nij', psi, grads[:, :, 1])

#     # Index arrays & sparse Bx, By
#     rowidx = np.repeat(tc, 3, axis=1).ravel()
#     colidx = np.tile(tf, (1, 3)).ravel()

#     B_x = sparse.csc_matrix(
#         (Bx_loc.ravel(), (rowidx, colidx)),
#         shape=(Np_coarse, Np_fine)
#     )
#     B_y = sparse.csc_matrix(
#         (By_loc.ravel(), (rowidx, colidx)),
#         shape=(Np_coarse, Np_fine)
#     )

#     return B_x, B_y

def calculate_pressure_B(p_fine, t_fine, p_coarse, t_coarse):

    Nt_fine   = t_fine.shape[0]
    Np_fine   = p_fine.shape[0]
    Nt_coarse = t_coarse.shape[0]
    Np_coarse = p_coarse.shape[0]

    fine_to_coarse = np.repeat(np.arange(Nt_coarse), 4)[:Nt_fine]

    tf = t_fine[:, :3]
    tc = t_coarse[fine_to_coarse, :3]
    pf = p_fine[tf]                          # (Nt_fine, 3, 2)

    y3_y1 = pf[:, 2, 1] - pf[:, 0, 1]
    y2_y1 = pf[:, 1, 1] - pf[:, 0, 1]
    x3_x1 = pf[:, 2, 0] - pf[:, 0, 0]
    x2_x1 = pf[:, 1, 0] - pf[:, 0, 0]

    grad_hat = np.array([[-1., -1.],
                         [ 1.,  0.],
                         [ 0.,  1.]])        # (3, 2), cols = [d/dξ1, d/dξ2]

    area_grad_x = 0.5 * (y3_y1[:, None] * grad_hat[:, 0]
                        - y2_y1[:, None] * grad_hat[:, 1])

    area_grad_y = 0.5 * (x2_x1[:, None] * grad_hat[:, 1]
                        - x3_x1[:, None] * grad_hat[:, 0])

    psi_children = np.array([
        [2/3, 1/6, 1/6],
        [1/6, 2/3, 1/6],
        [1/6, 1/6, 2/3],
        [1/3, 1/3, 1/3],
    ])

    psi = np.tile(psi_children, (Nt_coarse, 1))[:Nt_fine]

    Bx_loc = -np.einsum('nj,ni->nji', psi, area_grad_x)
    By_loc = -np.einsum('nj,ni->nji', psi, area_grad_y)

    rowidx = np.repeat(tc, 3, axis=1).ravel()
    colidx = np.tile(tf, (1, 3)).ravel()

    B_x = sparse.csc_matrix(
        (Bx_loc.ravel(), (rowidx, colidx)), shape=(Np_coarse, Np_fine))
    B_y = sparse.csc_matrix(
        (By_loc.ravel(), (rowidx, colidx)), shape=(Np_coarse, Np_fine))

    return B_x, B_y

#_______________________________________________________________________________________________________________________________________________________________

def calculate_pressure_stiffness_Kp(p_coarse, t_coarse):
    """Scalar pressure Laplacian K_p, reusing the existing stiffness assembler."""
    return calculate_velocity_A(p_coarse, t_coarse, kinematic_viscosity=1.0)

#_______________________________________________________________________________________________________________________________________________________________

def calculate_stabilization_Cp(p_coarse, t_coarse, gamma=0.05):
    Kp = calculate_pressure_stiffness_Kp(p_coarse, t_coarse)

    # element diameters (longest edge of each coarse triangle)
    v0, v1, v2 = p_coarse[t_coarse[:,0]], p_coarse[t_coarse[:,1]], p_coarse[t_coarse[:,2]]
    e01 = np.linalg.norm(v1-v0, axis=1)
    e12 = np.linalg.norm(v2-v1, axis=1)
    e20 = np.linalg.norm(v0-v2, axis=1)
    h_avg = np.mean(np.concatenate([e01, e12, e20]))

    Cp = gamma * h_avg**2 * Kp
    return Cp

#_______________________________________________________________________________________________________________________________________________________________

# def calculate_Saddle_point_K(A, B_x, B_y):
#     """Calculates the Saddle-Point matrix **K**"""

#     Np_coarse = B_x.shape[0]
#     Zero_pp = sparse.csc_matrix((Np_coarse, Np_coarse))

#     K_mat = bmat([
#         [A,       None,    B_x.T],
#         [None,    A,       B_y.T],
#         [B_x,     B_y,     Zero_pp]
#     ], format='csc')

#     return K_mat

def calculate_Saddle_point_K(A, B_x, B_y, Cp=None):
    Np_coarse = B_x.shape[0]

    if Cp is None:
        Zero_pp = sparse.csc_matrix((Np_coarse, Np_coarse))
        pp_block = Zero_pp
        
    else:
        pp_block = -Cp

    K_mat = bmat([
        [A,    None, B_x.T],
        [None, A,    B_y.T],
        [B_x,  B_y,  pp_block]
    ], format='csc')

    return K_mat

#_______________________________________________________________________________________________________________________________________________________________

def calculate_Neumann_BCs():
    return

#_______________________________________________________________________________________________________________________________________________________________

def calculate_Dirichlet_BCs(A_mat, lifting_function):
    lf_x, lf_y = lifting_function
    return A_mat @ lf_x, A_mat @ lf_y

#_______________________________________________________________________________________________________________________________________________________________

def calculate_pressure_lifting(B_x, B_y, lifting_function):
    lf_x, lf_y = lifting_function
    return B_x @ lf_x + B_y @ lf_y

#_______________________________________________________________________________________________________________________________________________________________

def calculate_F(A_mat, B_x, B_y, lifting_function):
    F_x, F_y = calculate_Dirichlet_BCs(A_mat, lifting_function)
    p = calculate_pressure_lifting(B_x, B_y, lifting_function)
    return np.concatenate([-F_x, -F_y, p])

#_______________________________________________________________________________________________________________________________________________________________

def evalOnTrigs(p, t, viscosity_function): # by Stefan Takacs
    
    Nt = t.shape[0]
    result = np.zeros(Nt,'d')
    for i in range(Nt):
        q = ( p[t[i,0]] + p[t[i,1]] + p[t[i,2]] ) / 3
        result[i] = viscosity_function(*q,*t[i,6:])

    return result

#_______________________________________________________________________________________________________________________________________________________________

def calculate_E_matrix(idx, dims):
    num_sensors = len(idx)
    
    data = np.ones(num_sensors)
    rows = np.arange(num_sensors)
    cols = idx
    
    return sparse.csc_matrix((data, (rows, cols)), shape=(num_sensors, dims))

#===============================================================================================================================================================
# SAVING/LOADING
#===============================================================================================================================================================

def save_simulation_data(p_fine, e_fine, t_fine, 
                         p_coarse, e_coarse, t_coarse,
                         ux, uy, p_sol,
                         name:str='SIM_n'):
    """Saves the solution and grid into a single file in compressed ```.npz``` format."""
    
    np.savez_compressed(f'Solutions/{name}.npz',
                        p_fine=p_fine,                        
                        e_fine=e_fine,
                        t_fine=t_fine,
                        p_coarse=p_coarse,                        
                        e_coarse=e_coarse,
                        t_coarse=t_coarse,
                        ux=ux,
                        uy=uy,
                        p_sol=p_sol)
    
    print(f"Simulation '{name}' data saved.")  

#_______________________________________________________________________________________________________________________________________________________________

def load_simulation_data(file_path:str='Solutions/Exchanger_device.npz'):
    """Loads the data from the compressed ```.npz``` format."""

    data = np.load(file_path)

    p_fine = data['p_fine']    
    e_fine = data['e_fine']
    t_fine = data['t_fine']

    p_coarse = data['p_coarse']
    e_coarse = data['e_coarse']
    t_coarse = data['t_coarse']

    ux = data['ux']
    uy = data['uy']
    p_sol = data['p_sol']

    return p_fine, e_fine, t_fine, p_coarse, e_coarse, t_coarse, ux, uy, p_sol
