"""Reconstruct wetting and mechanics from sparse witness data across geometry."""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesRegressor

from gel_reactive_port_timing import read_json, sha256
from gel_reactive_two_read_assay import write_new
from learned_hydraulic_closure import advance, fit_face_law
from patch_exchange_forward import _hydraulic_path


ROOT = Path(__file__).resolve().parent
SAMPLE = .2
STEPS = 25
WITNESS = .65
BONDS = (.32, .85)
ARMS = ('forest', 'constant_law', 'learned_law')
TASKS = {'forecast': (2, 4), 'interpolation': (2, 16)}
SOURCES = ['hydraulic_process_transfer_study.py', 'learned_hydraulic_closure.py', 'patch_exchange_forward.py',
           'patch_reservoir_joint_bank.py', 'patch_reservoir_face_bank.py', 'patch_reservoir_service.py',
           'gel_reactive_port_timing.py', 'gel_reactive_two_read_assay.py']


def register():
    now = datetime.now(timezone.utc)
    run = Path('runs')/now.strftime('hydraulic_process_transfer_%Y%m%dT%H%M%S%fZ')
    run.mkdir()
    roles = {'training': [48, 6384452005], 'test_geometry': [36, 6384599141], 'test_mechanism': [48, 6384692903]}
    plan = {'registered_utc': now.isoformat(), 'registered_shanghai': now.astimezone(timezone(timedelta(hours=8))).isoformat(),
            'roles': roles, 'source': {name:sha256(ROOT/name) for name in SOURCES},
            'kind': 'Synthetic process component benchmark without experimental samples or joint electrical endpoints.',
            'question': 'Can a learned positive hydration-dependent transport law turn sparse witness data into '
              'useful space, interpolation and forecasting predictions in a separate geometry?',
            'train': 'High-resolution synthetic fields allowed to all models for the same 48 mothers. Fit the nonconstant '
              'law from cumulative face mass balance. For forest use the matching noisy early packets and solution fields.',
            'held': 'Grouped fresh trajectories from SSPRK3 72-cell variable-diffusion solver. Geometry transfer uses '
              'thickness scales .78 and 1.22 outside train. Extra regime holds more severe dry transport and contact delay '
              'outside fitting law. All arms receive thickness, nominal material diffusivity, commanded film and '
              'two noisy water averages from the same thin witness. No actual held hydraulic coefficients or full fields.',
            'budget': 'Witness scale .65. First read .4. A separate .8 packet for spatial and future tests, and a '
              'separate 3.2 packet for temporal interpolation. Gaussian error .016; same raw packet across methods.',
            'predicted_times': np.round(SAMPLE*np.arange(STEPS),10).tolist(),
            'predicted_fields': 'Twelve depth-cell means, then near/far layer mean modulus. Continue to 4.8.',
            'test_geometry': [0.78, .98, 1.22], 'train_geometry': [.86, 1., 1.14], 'films': [.42, .78],
            'law_reference_scales': [.60,.76,.92,1.08,1.24], 'film_reference_scales': [.82,1.,1.18],
            'mechanical_only': 'Six candidates, require install late back stiffness >= .91 and work front stiffness <= .79 throughout. '
               'Candidate lab choice and mass-closure setup remain separate from the joint electrical time-window problem.',
            'score': 'Mean field absolute error for witness-off-band at .8, between-read main fields, future main fields, '
              'and installed/wet layer traces. Also six-action mechanical-only ungated choice and all-action Brier. '
              'Report both held populations; primary shifted future MAE versus constant and forest with paired bootstrap.',
            'bootstrap_seed': 548210}
    write_new(run/'REGISTERED.json',plan)
    print(run.as_posix(),flush=True)


def groups(plan, split):
    size, seed = plan['roles'][split]
    rng = np.random.default_rng(seed)
    out = rng.uniform(size=(size,7))
    for i, (nominal, thickness, gain, dry, p, film, drift) in enumerate(out):
        shift = split == 'test_mechanism'
        case = (i, .08+.07*nominal, .85+.24*thickness, .83+.34*gain)
        latent = (.22+.19*dry, 2.+1.05*p, .74+.41*film, .16+.18*drift) if shift else (.37+.22*dry, .82+1.25*p, .91+.18*film, 0.)
        yield case, latent, rng.normal(0., .016, size=3)


def truth(case, parameters, scale, boundary, *, count=STEPS, dt=SAMPLE):
    _, nominal, size, gain = case
    r, p, film_factor, delay = parameters
    film = boundary*film_factor
    dx = size*scale/72
    dw = nominal*gain
    increments = int(np.ceil(dt*max(2*dw/dx**2,dw/dx**2+film/dx,sum(BONDS))/.85))
    # Model change for the held wet-interface condition: the first sample is dry for part of its interval.
    # A separate fine time grid supplies a conservative trajectory delayed on the same physical clock.
    fine = dt / 20
    fine_steps = max(1,int(np.ceil(increments/20)))
    h, e, _ = _hydraulic_path(72,(count-1)*20+1,fine_steps,fine,dx,dw,(r,1.,p),BONDS,film,1.)
    index = np.maximum(0,np.arange(count)*20-int(np.rint(delay/fine)))
    return h[index], e[index]


def compress(a, solid):
    return np.column_stack((a.reshape(len(a),12,-1).mean(axis=-1),solid))


def middle(path, indices):
    a = path[indices]
    water = a[:, :-2]
    return water[:,water.shape[-1]//4:water.shape[-1]//2].mean(axis=-1)


def sample_inputs(plan, split):
    return np.array([[case[0],case[1],case[2], *witness_signals(case, z, eps)]
                     for case, z, eps in groups(plan, split)])


def witness_signals(case, parameters, noise):
    # Use only times .4, .8 and 3.2, with geometry and clocks identical across estimators.
    path, solid = truth(case, parameters, WITNESS, .55, count=17)
    return middle(compress(path,solid),[2,4,16])+noise


def targets(plan, split):
    scales = plan['train_geometry' if split == 'training' else 'test_geometry']
    full, witness = [], []
    for case, latent, _ in groups(plan, split):
        full.append([compress(*truth(case,latent,s,f)) for s in scales for f in plan['films']])
        witness.append(compress(*truth(case,latent,WITNESS,.55)))
    return np.asarray(full), np.asarray(witness)


def train_models(plan):
    first, full, w = sample_inputs(plan,'training'), *targets(plan,'training')
    records = []
    for case, latent, _ in groups(plan,'training'):
        profile, _ = truth(case,latent,1.,.78,count=101,dt=.02)
        records.append((profile,case[2]/72,.02,case[1]))
    by_wetting = sorted(records, key=lambda r:r[0][-1].mean())
    curves = np.array([fit_face_law(by_wetting[i:i+16]) for i in (0,16,32)])
    shapes = full.shape
    work = {}
    for task, indices in TASKS.items():
        x = model_inputs(first,indices)
        geometry = geometry_inputs(x,plan,'training')
        models = []
        for features, y in ((geometry,full.reshape(-1,np.prod(shapes[2:]))),(x,w.reshape(len(x),-1))):
            tree = ExtraTreesRegressor(n_estimators=112, min_samples_leaf=2, random_state=16221, n_jobs=6)
            models.append(tree.fit(features,y))
        work[task] = models
    return work,curves


def geometry_inputs(x, plan, role):
    scales = plan['train_geometry' if role=='training' else 'test_geometry']
    return np.asarray([[*row,s,f] for row in x for s in scales for f in plan['films']])


def model_inputs(reads, indices):
    tail = 4 if indices[1]==4 else 5
    return reads[:,[1,2,3,tail]]


def mechanical(paths):
    support = (.91 - paths[..., :10, -1].min(axis=-1))/.25
    soft = (paths[..., 16:, -2].max(axis=-1) - .79)/.25
    return np.maximum(support,soft)


def forest_estimate(models, x, plan, role):
    actual = [geometry_inputs(x,plan,role),x]
    tree_means = []
    outputs = []
    for i, (model, matrix) in enumerate(zip(models,actual,strict=True)):
        candidates = 6 if i==0 else 1
        sims = np.asarray([r.predict(matrix).reshape(-1,candidates,STEPS,14) for r in model.estimators_],dtype=np.float32)
        tree_means.append(sims.mean(axis=0))
        if i==0:
            margins = mechanical(sims)
            outputs.append((margins>0).mean(axis=0))
            outputs.append(margins.mean(axis=0))
    return tree_means[0],tree_means[1][:,0],*outputs


def physical_estimate(plan, reads, curves, *, matched_constant=False):
    scales = plan['law_reference_scales']
    if matched_constant:
        scales = np.linspace(scales[0], scales[-1], 3*len(scales))
    settings = [(curve+np.log(s),d) for curve in curves for s in scales
                                     for d in plan['film_reference_scales']]
    poses = [(s,f) for s in plan['test_geometry'] for f in plan['films']]
    result = {task:[] for task in TASKS}
    for _, d0, thickness, *obs in reads:
        bank, witnesses = [], []
        for curve, offset in settings:
            def path(scale, film):
                return advance(12,STEPS,SAMPLE,thickness*scale,d0,curve,film*offset,1.,BONDS)
            bank.append([path(s,f) for s,f in poses])
            witnesses.append(path(WITNESS,.55))
        fields, w = np.asarray(bank), np.asarray(witnesses)
        probabilities = mechanical(fields)
        for task, (a,b) in TASKS.items():
            signal = np.array([middle(z,[a,b]) for z in w])
            actual = [obs[0],obs[1 if b==4 else 2]]
            weights = softmax(-.5*np.square((actual-signal)/.016).sum(axis=1))
            def mean(values):
                return np.tensordot(weights,values,axes=1)
            result[task].append((mean(fields),mean(w),mean(probabilities>0),mean(probabilities)))
    return {task:tuple(np.asarray(r) for r in zip(*value,strict=True)) for task,value in result.items()}


def freeze(plan, run):
    models, curves = train_models(plan)
    items = {'learned_curves':curves}
    for role in ('test_geometry','test_mechanism'):
        reads = sample_inputs(plan,role)
        items[role+'_inputs'] = reads
        bank = {'learned_law':physical_estimate(plan,reads,curves),
                'constant_law':physical_estimate(plan,reads,np.zeros((1,5)),matched_constant=True)}
        for task, indices in TASKS.items():
            forecast = {k:v[task] for k,v in bank.items()}
            forecast['forest'] = forest_estimate(models[task],model_inputs(reads,indices),plan,role)
            for k, v in forecast.items():
                for label, array in zip(('paths','witness','risk','margin'),v,strict=True):
                    items['_'.join((role,task,k,label))] = array
        print(role, len(reads),flush=True)
    with (run/'FROZEN.npz').open('xb') as f:
        np.savez_compressed(f, **items)


def batch_margin(prob, margin):
    return np.lexsort((margin,prob),axis=1)[:,0]


def score(plan, run):
    predictions = np.load(run/'FROZEN.npz')
    outputs, rows = {}, {}
    for role in ('test_geometry','test_mechanism'):
        full,w = targets(plan,role)
        rows[role+'_witness'] = w
        rows[role+'_paths'] = full
        real_margin = mechanical(full)
        record = {}
        for method in ARMS:
            def data(task, typ):
                return predictions['_'.join((role,task,method,typ))]
            forecast = data('forecast','paths')
            difference = abs(forecast - full)
            i = list(range(3))+list(range(6,12))
            spatial = np.abs(data('forecast','witness')[:,4,i] - w[:,4,i]).mean(axis=1)
            interp = np.abs(data('interpolation','paths')[:,:,3:16,:12] - full[:,:,3:16,:12]).mean(axis=(1,2,3))
            future = difference[:,:,5:,:12].mean(axis=(1,2,3))
            mechanics = difference[:,:,5:,12:].mean(axis=(1,2,3))
            probability = data('forecast','risk')
            choice = batch_margin(probability,data('forecast','margin'))
            observed = (real_margin[np.arange(len(real_margin)),choice]<=0)
            brier = ((real_margin>0)-probability)**2
            values = {'spatial_mae':spatial,'interpolation_mae':interp,'future_mae':future,'mechanical_mae':mechanics,
                      'mechanical_issued_success':observed,'brier':brier.mean(axis=1)}
            record[method] = {name:float(x.mean()) for name,x in values.items()}
            for name,value in values.items():
                rows[f'{role}_{method}_{name}'] = np.asarray(value)
        rng = np.random.default_rng(plan['bootstrap_seed'])
        idx = rng.integers(len(full),size=(5000,len(full)))
        comparisons = {}
        for control in ('forest','constant_law'):
            delta = rows[f'{role}_learned_law_future_mae'] - rows[f'{role}_{control}_future_mae']
            comparisons[control] = {'difference':float(delta.mean()),'ci95':np.quantile(delta[idx].mean(axis=1),[.025,.975]).tolist()}
        outputs[role] = {'groups':len(full),'available_mechanical_choices':int((real_margin.min(axis=1)<=0.).sum()),
                         'records':record,'future_mae_paired':comparisons}
    with (run/'TRUTH_AND_PER_MOTHER.npz').open('xb') as f:
        np.savez_compressed(f,**rows)
    write_new(run/'SCORED.json',{'protocol_sha256':sha256(run/'REGISTERED.json'),'frozen_sha256':sha256(run/'FROZEN.npz'),
               'scores':outputs,'completed_utc':datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','forecast','score'))
    parser.add_argument('run',nargs='?',type=Path)
    a = parser.parse_args()
    if a.phase == 'register':
        register()
        return
    plan = read_json(a.run/'REGISTERED.json')
    for name, expected in plan['source'].items():
        if sha256(ROOT/name)!=expected:
            raise ValueError(f'Changed registered version of {name}.')
    if a.phase == 'forecast':
        freeze(plan,a.run)
        write_new(a.run/'FORECAST_RECEIPT.json', {'protocol_sha256':sha256(a.run/'REGISTERED.json'),
                  'frozen_sha256':sha256(a.run/'FROZEN.npz'),'completed_utc':datetime.now(timezone.utc).isoformat()})
    else:
        if read_json(a.run/'FORECAST_RECEIPT.json')['frozen_sha256']!=sha256(a.run/'FROZEN.npz'):
            raise ValueError('Scoring requires a valid forecast freeze.')
        score(plan,a.run)


if __name__ == '__main__':
    main()
