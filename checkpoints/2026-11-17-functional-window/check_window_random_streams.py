"""Check all RNG addresses used by the mixed-cohort witness and outcome drivers."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def addresses(plan):
    keys=plan['seeds']
    sensor=plan['spatial_probe']['snapshot_noise_seed']
    hydraulic=plan['spatial_probe']['shift_variation_seed']
    for role, rule in plan['event_roles'].items():
        n=rule['groups']
        yield rule['nominal_seed']+772,f'{role}:nominal parent sequence'
        yield rule['shift_seed']+772,f'{role}:exchange parent sequence'
        yield rule['mechanism_seed'],f'{role}:mixing Bernoulli sequence'
        for i in range(n):
            for branch in ('test','shift_test'):
                yield sensor+rule['spatial_noise_salt']+keys[branch]+i,f'{role}:{branch}:{i}:witness noise'
            yield hydraulic+rule['hydraulic_salt']+i,f'{role}:exchange:{i}:hydraulic variation'
            if role=='hidden_bath_stress':
                yield plan['hidden_bath_stress']['seed']+i,f'{role}:{i}:counterfactual bath law'


def preflight(plan):
    known={}
    for seed,use in addresses(plan):
        if seed in known:
            raise ValueError(f'Duplicate random-stream address {seed} for {use} and {known[seed]}.')
        known[seed]=use
    return known


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('protocol',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    encoded=args.protocol.read_bytes()
    found=preflight(json.loads(encoded))
    data={'protocol_sha256':hashlib.sha256(encoded).hexdigest(),
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'checked_utc':datetime.now(timezone.utc).isoformat(),'distinct_addresses':len(found),
          'scope':'All possible Bernoulli branch witness addresses, plus parent, mechanism, hydraulic and pressure streams.'}
    with args.output.open('x',encoding='utf8') as f:
        json.dump(data,f,indent=2)
    print(f'{len(found)} distinct generator addresses.')


if __name__=='__main__':
    main()
