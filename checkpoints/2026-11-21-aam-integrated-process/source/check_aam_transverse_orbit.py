"""Check the physical translation and the camera likelihood without fitting."""
import argparse
from pathlib import Path

import numpy as np

from aam_first_interval_paired_fork import future_for_action
from aam_formation_cycle_data import _read6, check_actual_state
from aam_formation_cycle_driver import read,sha,write
from aam_formation_local_data import formation_prefix_local
from aam_formation_local_initial import clone_initial,draw_initial
from aam_formation_path_cycle import stamp
from aam_snapshot_process import observe
from aam_transverse_orbit import camera_orbit,hidden_q_orbit,posterior_from_orbits,translate_packet
from package_aam_formation_local_cycle import source_closure


def same(a,b,y,box,wrap):
    pa, pb = a[0], b[0]
    expected=pa.points.copy()
    expected[:,1]=(expected[:,1]+y)%box
    error=pb.points-expected
    error -= np.rint(error/box)*box
    np.testing.assert_allclose(error,0.,rtol=0,atol=2e-11)
    for name in ('present','active','available','owner','edges','winding','counts'):
        np.testing.assert_array_equal(getattr(pa,name),getattr(pb,name))
    ne=pa.counts[1]
    edges=pa.edges[:ne]
    lp=pa.lifts[:ne].copy()
    lp[:,1]+=wrap[edges[:,1]]-wrap[edges[:,0]]
    np.testing.assert_array_equal(lp,pb.lifts[:ne])
    np.testing.assert_allclose(pa.events[:pa.counts[0],:10],pb.events[:pb.counts[0],:10],atol=2e-11,rtol=0.)
    return float(np.max(np.abs(error)))


def check(v5,out):
    info=v5/'REGISTERED.json'
    cfg=read(info)['config']
    source=source_closure([Path(__file__)])
    reg=out.with_name(out.stem+'_REGISTERED.json')
    offsets=np.asarray([0.1875,1.3125,3.,5.8125])
    write(reg,{**stamp(),'physics':str(info),'physics_sha256':sha(info),
          'sources':{str(p):sha(p) for p in source.values()},'offsets':offsets,
          'purpose':'one new engineering fixture only, no new scored mothers'})
    rng=np.random.default_rng(91041973)
    factors=np.asarray(cfg['factors'])
    base=draw_initial(cfg,rng)
    native=rng.integers(0,2**32,3,dtype=np.uint32)
    original=formation_prefix_local(cfg,base,factors,native_seeds=native,mother_id=0)
    seed=rng.integers(0,2**32,3,dtype=np.uint32)
    inputs={'native_seeds':seed,'placement_seed':4337112,'seed':998801}
    one=future_for_action(original['packet'],cfg['actions'][5],cfg,inputs,0)
    if not one['valid']:
        raise ValueError('Native original fixture failed.')
    box=cfg['motion']['box']
    for k,frame in enumerate(original['checkpoint_packets'][1:]):
        np.testing.assert_allclose(camera_orbit(frame,[0.])[0],original['clean'][k],atol=2e-12,rtol=0.)
    error=[]
    for dy in offsets:
        moved=clone_initial(base)
        wraps=np.floor((base['points'][:,1]+dy)/box).astype(int)
        moved['points'][:,1]=(base['points'][:,1]+dy)%box
        others=formation_prefix_local(cfg,moved,factors,native_seeds=native,mother_id=0)
        if not others['valid']:
            raise ValueError('Shifted fixture did not create three frames.')
        for a,b in zip(original['checkpoint_packets'],others['checkpoint_packets'],strict=True):
            error.append(same(a,b,dy,box,wraps))
        for j,a in enumerate(original['checkpoint_packets'][1:]):
            shifted=translate_packet(a,dy)
            check_actual_state(shifted[0])
            np.testing.assert_allclose(camera_orbit(a,[dy])[0],observe(shifted),atol=5e-13,rtol=0.)
            np.testing.assert_allclose(camera_orbit(a,[dy])[0],others['clean'][j],atol=2e-11,rtol=0.)
        second=future_for_action(others['packet'],cfg['actions'][5],cfg,inputs,0)
        if not second['valid']:
            raise ValueError('Shifted action could not proceed.')
        for a,b in zip(one['snapshots'],second['snapshots'],strict=True):
            x,y=a['packet'],b['packet']
            error.append(same(x,y,dy,box,wraps))
            np.testing.assert_allclose(hidden_q_orbit(x,[dy])[0],_read6(y,b['properties'])[:3],atol=2e-11,rtol=0.)
            np.testing.assert_allclose(_read6(x,a['properties'])[3:],_read6(y,b['properties'])[3:],atol=2e-10,rtol=0.)
    camera=np.stack([camera_orbit(p,offsets) for p in original['checkpoint_packets'][1:]],axis=1)[None]
    posterior=posterior_from_orbits(camera,original['clean'],cfg['noise'])
    reference=np.exp(-.5*np.sum(((camera-original['clean'])/np.asarray(cfg['noise']))**2,axis=(-2,-1)))
    np.testing.assert_allclose(posterior,reference/reference.sum(),atol=1e-13,rtol=0.)
    write(out,{**stamp(),'passed':True,'registration_sha256':sha(reg),'controlled_parity_frames':len(error),
              'maximum_coordinate_error':max(error),'new_scored_mothers':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v5',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    check(a.v5.resolve(),a.output.resolve())
