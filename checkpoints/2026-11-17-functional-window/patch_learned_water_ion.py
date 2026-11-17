"""Propagate the learned hydration law through the unchanged conservative ion equation."""

import numpy as np

from learned_hydraulic_closure import advance
from patch_reservoir_joint_bank import _joint_ion_trace, prepared_fields


def joint_trace(parent, actions, cfg, bath, kinetics, shape_modes, cells, step, times, curve):
    dx = parent['thickness']/cells
    nominal = parent['water_D']
    ticks = np.rint(times/step).astype(int)
    if not len(ticks) or ticks[0] != 0 or not np.allclose(ticks*step,times,atol=1e-12,rtol=0):
        raise ValueError('Use registered output clocks aligned to the ion solver.')
    transfers, channel = np.unique([a['wet_transfer'] for a in actions],return_inverse=True)
    rates, gamma, loss, interface = kinetics.T
    cfl = step*max(4*cfg['mobile_D']/dx**2+loss.max()/dx,rates.max())
    if cfl>2.4:
        raise ValueError('Reduce the integration step before propagating mobile ions.')
    bonds = (cfg['bond_on'],cfg['bond_off'])
    fields = np.array([advance(cells,2*ticks[-1]+1,step/2,parent['thickness'],nominal,curve,k,bath,bonds) for k in transfers])
    h = np.ascontiguousarray(fields[...,:cells])
    face_d = nominal*np.exp(np.interp(h[...,0],np.linspace(0.,1.,len(curve)),curve))
    ratio = transfers[:,None]*dx/(2.*face_d)
    surface = (h[...,0]+ratio*bath)/(1.+ratio)
    mobile = cfg['mobile_D']*(.10 + .90*h)
    resistance = dx/(2.*mobile[...,0])
    faces = 2./(1./mobile[...,:-1]+1./mobile[...,1:])/dx**2
    prepared = prepared_fields(parent,actions,cfg,shape_modes,cells)
    constants = (cfg['dry_conductivity'],cfg['water_background'],cfg['ion_conductivity'])
    records = np.empty((len(kinetics),len(ticks),len(actions),4))
    records[...,0] = h[:,2*ticks].mean(axis=-1).T[:,channel]
    records[...,1:3] = fields[:,2*ticks,-2:].transpose(1,0,2)[:,channel,:]
    closure = 0.
    for i, (k,p,outgoing,rho) in enumerate(zip(rates,gamma,loss,interface,strict=True)):
        ion_faces = faces.copy()
        ion_faces[...,cells//2-1] /= 1.+rho*parent['thickness']*dx/cfg['mobile_D']*ion_faces[...,cells//2-1]
        values, _, error, minimum = _joint_ion_trace(h**p,h,surface,resistance,ion_faces,prepared[i:i+1],channel,
            ticks,np.array([[k,outgoing]]),dx,step,constants,False)
        if minimum<-1e-10 or error>2e-11*max(1.,parent['thickness']):
            raise ArithmeticError('The coupled prediction violated nonnegative inventory or mass balance.')
        records[i,...,3] = values[0]
        closure = max(closure,error)
    return records, np.ascontiguousarray(h[channel][:,2*ticks]), closure
