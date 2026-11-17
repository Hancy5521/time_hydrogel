"""Replay held process scores and independently integrate the learned finite-volume law."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from learned_hydraulic_closure import advance


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def independent_path(learned):
    count, dx, nominal, film = 12, .087, .12, .55
    clock = .2*np.arange(25)

    def balances(_, state):
        h, b = state[:count], state[count:2*count]
        d = nominal*np.exp(np.interp(h,np.linspace(0,1,5),learned))
        flux = np.zeros(count+1)
        flux[0] = (1-h[0])/(1/film+.5*dx/d[0])
        flux[1:-1] = (h[:-1]-h[1:])/(.5*dx*(1/d[:-1]+1/d[1:]))
        hyd = (flux[:-1]-flux[1:])/dx
        bond = .32*(1-h)*(1-b) - .85*h*b
        return np.r_[hyd,bond,flux[0]]

    ref = solve_ivp(balances,(0.,clock[-1]),np.r_[np.zeros(count),np.ones(count),0.],
                    method='Radau',rtol=2e-9,atol=1e-11,t_eval=clock)
    if not ref.success:
        raise ArithmeticError(ref.message)
    width = count*dx
    out = advance(count,len(clock),.2,width,nominal,learned,film,1.,(.32,.85))
    h, b = ref.y[:count].T, ref.y[count:2*count].T
    stiffness = ((.27+.93*b)*(1-.27*h)).reshape(len(clock),2,-1).mean(axis=-1)
    reference = np.column_stack((h,stiffness))
    return {'max_absolute_stepper_difference':float(abs(out-reference).max()),
            'max_radau_water_inventory_closure':float(abs(dx*h.sum(axis=-1)-ref.y[-1]).max())}


def window_violation(process):
    # Determine signed mechanical limits directly by traversing the two windows.
    return max(max((.91-f[-1])/.25 for f in process[:10]),
               max((f[-2]-.79)/.25 for f in process[16:]))


def replay(predicted,observed,summary):
    maximum = 0.
    for group, receipt in summary.items():
        field, witness = observed[group+'_paths'], observed[group+'_witness']
        reference = np.array([[window_violation(case) for case in slate] for slate in field])
        for name, target in receipt['records'].items():
            fp = predicted[group+'_forecast_'+name+'_paths']
            fw = predicted[group+'_forecast_'+name+'_witness']
            ip = predicted[group+'_interpolation_'+name+'_paths']
            risk = predicted[group+'_forecast_'+name+'_risk']
            margin = predicted[group+'_forecast_'+name+'_margin']
            chosen = [min(range(6),key=lambda k:(prob[k],m[k],k)) for prob,m in zip(risk,margin,strict=True)]
            replayed = {
                'spatial_mae':float(np.abs(fw[:,4,[0,1,2,6,7,8,9,10,11]]-witness[:,4,[0,1,2,6,7,8,9,10,11]]).mean()),
                'interpolation_mae':float(np.abs(ip[:,:,3:16,:12]-field[:,:,3:16,:12]).mean()),
                'future_mae':float(np.abs(fp[:,:,5:,:12]-field[:,:,5:,:12]).mean()),
                'mechanical_mae':float(np.abs(fp[:,:,5:,12:]-field[:,:,5:,12:]).mean()),
                'brier':float(np.square(risk-(reference>0.)).mean()),
                'mechanical_issued_success':float(np.mean([r[i]<=0 for r,i in zip(reference,chosen,strict=True)])),
            }
            maximum = max(maximum,max(abs(replayed[k]-target[k]) for k in replayed))
    if maximum>5e-12:
        raise ArithmeticError(f'Score replay disagrees by {maximum}.')
    return maximum


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    protocol = a.run/'REGISTERED.json'
    p0 = json.loads(protocol.read_bytes())
    forecast_path = a.run/'FROZEN.npz'
    measured_path = a.run/'TRUTH_AND_PER_MOTHER.npz'
    score = a.run/'SCORED.json'
    scored = json.loads(score.read_bytes())
    frozen = json.loads((a.run/'FORECAST_RECEIPT.json').read_bytes())
    assert scored['frozen_sha256']==frozen['frozen_sha256']==digest(forecast_path)
    assert scored['protocol_sha256']==frozen['protocol_sha256']==digest(protocol)
    assert p0['registered_utc']<frozen['completed_utc']<scored['completed_utc']
    data, ref = np.load(forecast_path),np.load(measured_path)
    result = {'completed_utc':datetime.now(timezone.utc).isoformat(),'registered_sha256':digest(protocol),
              'frozen_sha256':digest(forecast_path),'score_sha256':digest(score),
              'score_replay_max_absolute_difference':replay(data,ref,scored['scores']),
              'learned_law_stepper_against_independent_Radau':independent_path(data['learned_curves'][1])}
    with a.output.open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(result)


if __name__=='__main__':
    main()
