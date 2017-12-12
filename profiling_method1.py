import numpy as np

q_vecs = np.random.random((10000, 3))
atom_vecs = np.random.random((100, 3))
 
@profile  # this bit is important!
def method1(q_vecs, atom_vecs):
    Nq = q_vecs.shape[0]
    ampsR = np.zeros( Nq ) 
    ampsI = np.zeros( Nq )
    for i_q, q in enumerate( q_vecs):
        qx,qy,qz = q
        for i_atom, atom in enumerate( atom_vecs):
            ax,ay,az = atom
            phase = qx*ax + qy*ay + qz*az
            ampsR[i_q] += np.cos( -phase)
            ampsI[i_q] += np.sin( -phase)
    I = ampsR**2 + ampsI**2 
    return I

method1(q_vecs, atom_vecs)