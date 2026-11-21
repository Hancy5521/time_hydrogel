"""Build a continuous-control continuation cache on actual large-atlas histories.

Import does not open data, sample, fit, or advance a simulator. The caller
registers the all-bank receipts, the fixed uniform sample and this worker
before running any native continuations.
"""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from aam_bernstein_feed_design import rank_thresholds
from aam_common_phase_hidden import history_orbit
from aam_edge_fork import State
from aam_formation_actual_rollout import packet_fingerprint
from aam_formation_cycle_driver import np_load, np_save, read, sha, write
from aam_formation_path_cycle import load_tree, stamp
from aam_phase_marginal_tape_pgas import phase_offsets
from aam_transverse_orbit import camera_orbit, hidden_q_orbit
from audit_aam_endpoint_tape_move import artifact, generating_state
from run_aam_continuous_inverse import TIMES, original_losses, path_block


NAMESPACE = 'aam-original-large-atlas-384-phase-continuations-v1'
BANK_NAMESPACE = 'aam-large-shared-phase-original-bank-v1'
BANK_N = 8192
BANK_BLOCK_SIZE = 32
PHASE_COUNT = 16
CACHE_N = 384
SOURCE_UNITS = tuple(range(2026145000, 2026145000 + BANK_N))
SELECTION_WORDS = (20261121, 43541, BANK_N, CACHE_N)
FUTURE_ROLE = 3
FUTURE_REPEAT = 0
NATIVE_CALLS_PER_MOTHER = len(TIMES) * 7 * 3


def select_bank_rows():
    """The immutable row roster, sampled uniformly without replacement."""
    generator = np.random.default_rng(np.random.SeedSequence(SELECTION_WORDS))
    return np.sort(generator.choice(BANK_N, size=CACHE_N, replace=False))


def bank_input_boundary(bank_root):
    """Verify only inputs in the original bank role and describe the new cache."""
    origin = Path(bank_root).resolve(strict=True)
    registered, acquired, frozen, audit = (origin / f'{key}.json' for key in
                                          ('REGISTERED', 'BANK_ACQUIRED', 'BANK_FROZEN', 'INDEPENDENT_AUDIT'))
    reg, acq, lock, cert = map(read, (registered, acquired, frozen, audit))
    registered_hash, acquired_hash = sha(registered), sha(acquired)
    if (reg['namespace'] != BANK_NAMESPACE or reg['units']['bank'] != list(SOURCE_UNITS)
            or reg['block'] != BANK_BLOCK_SIZE or reg['phases'] != PHASE_COUNT
            or acq['role'] != 'bank' or acq['native_calls'] != 3 * BANK_N
            or lock['acquired_sha256'] != acquired_hash or lock['bank_mothers'] != BANK_N
            or cert['passed'] is not True or cert['registration_sha256'] != registered_hash
            or cert['bank_mothers'] != BANK_N or cert['all_archive_hashes_verified'] is not True):
        raise ValueError('Require the original bank role and its complete independent audit.')
    arrays = artifact(lock['arrays'], origin / 'FULL_BANK_ORBITS.npz')
    blocks = acq['rows']
    if (len(blocks) != BANK_N // BANK_BLOCK_SIZE
            or tuple(unit for block in blocks for unit in block['units']) != SOURCE_UNITS):
        raise ValueError('Frozen bank mothers or raw block order changed.')

    records, raw_hashes = [], {}
    for i in map(int, select_bank_rows()):
        b, k = divmod(i, BANK_BLOCK_SIZE)
        block = blocks[b]
        block_units = list(SOURCE_UNITS[i-k:i-k+BANK_BLOCK_SIZE])
        if block['units'] != block_units or block['native_calls'] != 3 * BANK_BLOCK_SIZE:
            raise ValueError('Original block does not contain all registered independent trajectories.')
        source = origin / 'bank' / f'{block_units[0]}_RAW.npz'
        if str(source) not in raw_hashes:
            artifact(block['raw'], source)
            raw_hashes[str(source)] = block['raw']['sha256']
        records.append({'row': i, 'unit': SOURCE_UNITS[i], 'in_block': k, 'block_units': block_units,
                        'raw_block': block['raw']})
    hashes = {str(registered): registered_hash, str(acquired): acquired_hash,
              str(frozen): sha(frozen), str(audit): sha(audit), str(arrays): lock['arrays']['sha256']}
    return {'namespace': NAMESPACE, 'bank_root': str(origin), 'physics': reg['physics'],
            'bank_arrays': lock['arrays'], 'selection_seed_words': list(SELECTION_WORDS),
            'selected_rows': [record['row'] for record in records], 'selected_records': records,
            'n_independent_mothers': len(records), 'n_selected_raw_blocks': len(raw_hashes),
            'phase_count': PHASE_COUNT, 'native_calls_per_selected_mother': NATIVE_CALLS_PER_MOTHER,
            'inputs': {**hashes, **raw_hashes}}


@lru_cache(maxsize=1)
def _bank_arrays(path, fingerprint, nbytes):
    reference = {'path': path, 'sha256': fingerprint, 'bytes': nbytes}
    result = np_load(artifact(reference, Path(path)))
    if (result['unit'].tolist() != list(SOURCE_UNITS) or result['local'].shape != (8192,16,3,2,3)
            or result['clean'].shape != (8192,16,3,4) or result['measured'].shape != (8192,3,4)):
        raise ValueError('Archived bank arrays differ from the original three-frame bank.')
    return result


@lru_cache(maxsize=2)
def _raw_block(path, fingerprint, nbytes):
    reference = {'path': path, 'sha256': fingerprint, 'bytes': nbytes}
    return load_tree(artifact(reference, Path(path)))['rows']


def load_selected_c3(bank_root, boundary, record, *, bank_arrays=None):
    """Read an exact original starting packet and verify its three-frame orbit."""
    r = record['raw_block']
    i, k, unit = (int(record[key]) for key in ('row', 'in_block', 'unit'))
    expected_block = list(SOURCE_UNITS[i-k:i-k+BANK_BLOCK_SIZE])
    origin = Path(bank_root).resolve(strict=True)
    source = origin / 'bank' / f'{expected_block[0]}_RAW.npz'
    if (boundary['namespace'] != NAMESPACE or origin != Path(boundary['bank_root'])
            or i not in boundary['selected_rows'] or SOURCE_UNITS[i] != unit or i % BANK_BLOCK_SIZE != k
            or record['block_units'] != expected_block or Path(r['path']) != source
            or boundary['inputs'][str(source)] != r['sha256']):
        raise ValueError('Selected route has no immutable original bank identity.')
    if bank_arrays is None:
        ref = boundary['bank_arrays']
        bank_arrays = _bank_arrays(ref['path'], ref['sha256'], ref['bytes'])
    rows = _raw_block(str(source), r['sha256'], r['bytes'])
    if [row['unit'] for row in rows] != expected_block or bank_arrays['unit'][i] != unit:
        raise ValueError('Raw full trajectories and compact original bank order disagree.')
    row = rows[k]
    measured, raw = row['measured'], row['raw']
    generating_state(boundary['physics'], unit, measured, raw)
    np.testing.assert_array_equal(measured, bank_arrays['measured'][i])
    initial, *observed = raw['packets']
    offsets = phase_offsets(initial)
    if len(observed) != 3 or len(offsets) != PHASE_COUNT:
        raise ValueError('Require the original three snapshots and associated 16 phase offsets.')
    cameras = np.stack([camera_orbit(packet, offsets) for packet in observed], axis=1)
    observed_origin = history_orbit(raw['packets'], offsets)
    np.testing.assert_allclose(cameras, bank_arrays['clean'][i], rtol=0., atol=2.e-13)
    np.testing.assert_allclose(observed_origin, bank_arrays['local'][i], rtol=0., atol=2.e-13)
    return observed[-1], offsets


def future_stream(unit):
    """Same seed domains as the original planning role, separate namespace.

    SeedSequence is addressed by four SHA-256 little-endian words from the
    new namespace, the source mother ID, role 3, repeat 0, and child 1/2/3.
    Only the planner's training cache uses these addresses.
    """
    entropy = np.frombuffer(sha256(NAMESPACE.encode()).digest()[:16], dtype='<u4').tolist()
    generators = [np.random.default_rng(np.random.SeedSequence([*entropy, int(unit), FUTURE_ROLE, FUTURE_REPEAT, c]))
                  for c in (1, 2, 3)]
    return {'seed': int(generators[0].integers(0, 2**32-51, dtype=np.uint32)),
            'native_seeds': generators[1].integers(0, 2**32, size=3, dtype=np.uint32),
            'placement_seed': int(generators[2].integers(0, 2**32, dtype=np.uint32))}


def phase_risks(raw_path, offsets, normalization):
    """Read all phase-specific losses from one native rank-strata archive.

    The phase orbit shifts the same future states. The periodic spring model
    and global conversion are invariant under translation in the ideal law.
    Only the fixed spatial apertures are reread, before evaluating loss.
    """
    data = np_load(raw_path)
    values, controls = data['values'], data['controls']
    shifts = np.asarray(offsets, dtype=np.float64)
    if (values.shape != (len(TIMES) * 7,3,6) or shifts.shape != (PHASE_COUNT,)
            or shifts[0] != 0. or not np.isfinite(shifts).all()
            or controls.shape != (len(TIMES) * 7,2)
            or not np.array_equal(controls[:, 0], np.repeat(TIMES, 7))
            or not np.all(data['completed'])):
        raise ValueError('Require every original time and seven legal placement-count futures.')
    source_material = data['source_material']
    geometry = data['source_geometry']
    expected = np.arange(PHASE_COUNT) * (geometry[2] / PHASE_COUNT)
    np.testing.assert_array_equal(shifts, expected)
    streams = rank_thresholds(int(data['placement_seed']))
    np.testing.assert_array_equal(controls[:, 1], np.tile(streams['thresholds'], len(TIMES)))

    alternatives = np.broadcast_to(values, (PHASE_COUNT, *values.shape)).copy()
    for k, control in enumerate(controls):
        material = source_material.copy()
        material[3] = control[0]
        for j in range(3):
            state = State(*(data[f'state_{key}'][k, j+1] for key in State._fields))
            packet = (state, material, geometry, data['source_rates'], data['source_factors'])
            alternatives[:, k, j, :3] = hidden_q_orbit(packet, shifts)
    np.testing.assert_allclose(alternatives[0], values, rtol=0., atol=2.e-11)
    phase_values = alternatives.reshape(PHASE_COUNT,len(TIMES),7,3,6)
    loss, blocks = original_losses(phase_values, normalization)
    original, _ = original_losses(values.reshape(len(TIMES),7,3,6), normalization)
    np.testing.assert_allclose(loss[0], original, rtol=0., atol=2.e-11)
    return {'values': phase_values, 'losses': loss, 'loss_blocks': blocks}


def cache_one(job):
    """Registered worker job = (output folder, input boundary, row, normalization)."""
    out, boundary, record, normalization = job
    out = Path(out).resolve(strict=True)
    unit = int(record['unit'])
    raw_path = out / f'{unit}_RAW.npz'
    compact_path = out / f'{unit}_PHASE_RISKS.npz'
    if raw_path.exists() or compact_path.exists():
        raise FileExistsError(f'Continuation cache already exists for mother {unit}.')
    stream = future_stream(unit)
    try:
        rank = rank_thresholds(stream['placement_seed'])
        controls = np.column_stack((np.repeat(TIMES, 7), np.tile(rank['thresholds'], len(TIMES))))
        with threadpool_limits(1):
            packet, offsets = load_selected_c3(boundary['bank_root'], boundary, record)
            fingerprint = packet_fingerprint(packet)
            _, raw = path_block(out, Path(raw_path.name), packet, controls, boundary['physics'], stream,
                                f'{NAMESPACE}:{unit}')
            risks = phase_risks(artifact(raw, raw_path), offsets, normalization)
            compact = np_save(compact_path, unit=unit, row=record['row'], offsets=offsets, **risks)
    except Exception as error:
        write(out / f'{unit}_CACHE_FAILED.json', {**stamp(), 'unit': unit, 'row': record['row'],
              'error': repr(error), 'stream': stream, 'raw_complete_at_failure': raw_path.is_file()})
        raise
    return {'unit': unit, 'row': record['row'], 'original_raw_block': record['raw_block'], 'c3_sha256': fingerprint,
            'native_calls': NATIVE_CALLS_PER_MOTHER, 'streams': stream, 'raw': raw, 'compact': compact}
