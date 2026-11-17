"""Independently replay frozen mixture decisions and check observation-only blending."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audit_patch_process_transfer import export, read_context, read, same_or_float, sha


SINGLE_LAWS = ('constant_physics', 'learned_physics')
EXTENDED = ('equal_mixture', 'evidence_mixture')


def fixture(args):
    import patch_process_mixture_study as process
    from reservoir_coupon_likelihood import Read

    protocol, plan, task, request, _ = read_context(args.run)
    obs = plan['probes']
    witness_clock = plan['spatial_probe']['times']
    early_assay_deadline = max(obs['last'], max(witness_clock), max(obs['hydration_times']))
    if early_assay_deadline >= request['install_end']:
        raise ValueError('A physical prediction reads material past its registration deadline.')
    # Independent, deliberately artificial measurements. No held observations or truth are loaded.
    rows = [
        [Read(obs[side], obs['last'], np.asarray([h, h+.03]), dose, dose, .0)
         for side, h, dose in zip(('low', 'high'), hydration, salts, strict=True)]
        for hydration, salts in (([.34, .53], [.13, .21]), ([.25, .42], [.20, .34]))
    ]
    dry = np.array([.45, .54])
    thin = np.array([[.41, -2.30], [.49, -1.76]])
    context = {'known': {'thickness': 1.02, 'water_D': .112, 'release': .86}}
    source = Path(plan['process_transfer']['hydraulic_freeze'])
    with np.load(source, allow_pickle=False) as states:
        learned = states['learned_curves'][1]
    with patch.object(process, 'packet', return_value=(rows, dry, None)):
        weight = process.branch_evidence(context, task, plan, thin, learned)
        flipped_hidden_label = process.branch_evidence({**context, 'shift': True}, task, plan, thin, learned)
        identical_laws = process.branch_evidence(context, task, plan, thin, np.zeros_like(learned))
    thin_shifted = thin.copy()
    thin_shifted[1] += (.06, .26)
    with patch.object(process, 'packet', return_value=([rows[0], rows[0]], np.array([dry[0], dry[0]]), None)):
        changed_second_state = process.branch_evidence(context, task, plan, thin_shifted, learned)
    if not np.array_equal(weight, flipped_hidden_label):
        raise ValueError('Hidden mechanism labels influence weights given identical observed data.')
    if not np.array_equal(identical_laws, np.array([.5, .5])):
        raise ValueError('Two identical candidate laws do not receive equal prior mass.')
    if weight[0] != changed_second_state[0]:
        raise ValueError('The second state influences the first state weighting.')
    export(args.output, {'passed': True, 'protocol_sha256': sha(protocol),
                         'hypothetical_weight': weight.tolist(), 'identical_law_weights': identical_laws.tolist(),
                         'after_perturbing_other_states_inputs': changed_second_state.tolist(),
                         'latest_assay_time': early_assay_deadline, 'installation_end': request['install_end'],
                         'scope': 'Only an observation-interface fixture. In held simulations the true mechanism generates the noisy early readings, without exposing the label to the likelihood calculation.'})


def score_groups(frozen, margin, truth, weights, plan, original):
    probabilities, decisions = {}, {}
    for p in (*SINGLE_LAWS, 'direct_trees'):
        probabilities[p] = frozen[p+'_probability']
        decisions[p] = frozen[p+'_choice']
    physical_means = [frozen[f'{p}_margin'] for p in SINGLE_LAWS]
    for name, learned_weight in zip(EXTENDED, (np.full_like(weights,.5), weights), strict=True):
        a = learned_weight[..., None]
        probabilities[name] = (1.-a)*probabilities[SINGLE_LAWS[0]]+a*probabilities[SINGLE_LAWS[1]]
        combined_mean = (1.-a)*physical_means[0]+a*physical_means[1]
        decisions[name] = (probabilities[name] + plan['risk_weight']*combined_mean).argmin(axis=-1)
    number, states, actions = margin.shape
    done = {}
    for label, choice in decisions.items():
        if choice.dtype.kind not in 'iu' or choice.min()<0 or choice.max()>=actions:
            raise ValueError('Frozen mixture action outside the common candidate set.')
        if not np.isfinite(probabilities[label]).all() or probabilities[label].min()<-1e-12 or probabilities[label].max()>1.+1e-12:
            raise ValueError('Invalid mixture probabilities.')
        selected = margin[np.arange(number)[:,None], np.arange(states)[None,:], choice]
        done[label] = (selected<=0.).mean(axis=1)
    for title, group in (('all', np.ones_like(truth)), ('nominal', ~truth), ('exchange', truth)):
        row = original[title]
        n = int(group.sum())
        if row['mothers']!=n or row['requests_with_feasible_action']!=int((margin[group].min(axis=-1)<=0).sum()):
            raise ValueError('The feasible or mother counts differ.')
        rng = np.random.default_rng(plan['mechanism_mixture']['bootstrap_seed'])
        draw = rng.integers(n, size=(5000,n))
        same_or_float(weights[group].mean(), row['mean_learned_law_weight'], title+'/weight')
        for name, scores in done.items():
            success = int(states*scores[group].sum())
            brier = ((probabilities[name][group] - (margin[group]>0.))**2).mean()
            same_or_float(success, row['methods'][name]['successes'], title+'/'+name+'/success')
            same_or_float(brier, row['methods'][name]['brier'], title+'/'+name+'/Brier')
            if name == 'evidence_mixture':
                continue
            diff = done['evidence_mixture'][group]-scores[group]
            claimed = row['evidence_mixture_differences'][name]
            same_or_float(diff.mean(), claimed['difference'], title+'/'+name+'/difference')
            same_or_float(np.quantile(diff[draw].mean(axis=1), [.025,.975]), claimed['ci95'], title+'/'+name+'/CI')
    return decisions


def score(args):
    protocol, plan, _task, _request, _clock = read_context(args.run)
    freeze, family, truth_npz = (args.run/name for name in ('FROZEN.npz', 'FAMILY_WEIGHTS.npz', 'TRUTH.npz'))
    receipt = read(args.run/'FORECAST_RECEIPT.json')
    scored = read(args.run/'SCORED.json')
    digest = sha(protocol)
    if (digest != receipt['protocol_sha256'] or digest != scored['protocol_sha256']
            or sha(freeze) != receipt['freeze_hashes']['FROZEN.npz'] or sha(family) != receipt['freeze_hashes']['FAMILY_WEIGHTS.npz']
            or sha(freeze) != scored['frozen_sha256'] or sha(family) != scored['family_sha256']):
        raise ValueError('Frozen mixture assets or protocol changed.')
    if not plan['registered_utc'] < receipt['completed_utc'] < scored['completed_utc']:
        raise ValueError('Registration and freeze did not precede scoring.')
    with np.load(freeze, allow_pickle=False) as a, np.load(family, allow_pickle=False) as b, np.load(truth_npz, allow_pickle=False) as c:
        names = c['mother']
        ys, shifted = c['margin'], c['exchange']
        weights = b['learned_family_weight']
        # Read only decision arrays, not the process paths embedded in the single-law archive.
        frozen = {k: a[k] for k in a.files if k.endswith(('_probability', '_choice', '_margin'))}
        if not np.array_equal(a['mother'], names) or len(names)!=len(set(names)):
            raise ValueError('Mixture truth mothers do not align to the forecast.')
    if shifted.dtype!=bool or weights.shape!=ys.shape[:2] or not np.isfinite(weights).all() or weights.min()<0 or weights.max()>1:
        raise ValueError('Invalid independent score inputs.')
    if len(names)!=plan['event_roles']['confirmation']['groups']:
        raise ValueError('Wrong evaluation population size.')
    decisions = score_groups(frozen, ys, shifted, weights, plan, scored['results'])
    # Pairwise definition of ROC AUC, with half credit for a tie.
    alpha = weights.mean(axis=1)
    compared = alpha[shifted,None]-alpha[~shifted]
    auc = (np.count_nonzero(compared>0)+.5*np.count_nonzero(compared==0))/compared.size
    same_or_float(auc, scored['mechanism_auc_audit_only'], 'mechanism AUC')
    export(args.output, {'passed': True, 'protocol_sha256': digest, 'freeze_sha256': sha(freeze),
                         'family_sha256': sha(family), 'released_truth_sha256': sha(truth_npz),
                         'scored_sha256': sha(args.run/'SCORED.json'), 'mothers':len(names),
                         'frozen_mixture_different_action_requests': int((decisions[EXTENDED[0]]!=decisions[EXTENDED[1]]).sum()),
                         'independent_checks': ['mixture ranking', 'per-arm Brier', 'ungated successes', 'paired mother intervals', 'mechanism AUC from ranks', 'hash and time ordering'],
                         'scope': 'Both pooled policies were reconstructed from frozen inputs. This audit does not refit trees, rerun PDEs, or re-estimate weights from raw measurement files.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fixture', 'score'))
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    (fixture if args.action == 'fixture' else score)(args)


if __name__ == '__main__':
    main()
