"""Complete V5 resources; no recursive historical ZIPs and no science on import.

Only an inactive complete24-mother run and its actual whole-raw audit PASS may
be packaged.  All current-version physical payloads and exact external inputs
are embedded.  Restore original paths using RESTORE_PATHS.json; no untested
arbitrary-path or dependency-free Python execution is claimed.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import zipfile


ROOT = Path(__file__).resolve().parent
DEFAULT_LOG = ROOT/'logs/research/20261119T014213543579Z'
NAMESPACE = 'aam-formation-explicit-initial-local-v5-20261119'
DOC_PREFIX = 'AAM_FORMATION_LOCAL_V5_'
ARMS = ('global_initial_smc64', 'local_initial_smc64', 'prior_sir3904')
REVIEW_FILES = ('WORK_V5_SCORED_MATH_PRE_REVIEW_FINAL_ONLY.txt',
                'WORK_V5_SCORED_MATH_PRE_REVIEW_SEND_RECEIPT.json',
                'V5_WORK_TECHNICAL_PRE_REVIEW_RECEIVED.json')


def stamp():
    at = datetime.now(timezone.utc)
    return {'actual_utc': at.isoformat(), 'asia_shanghai': at.astimezone(timezone(timedelta(hours=8))).isoformat()}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(path):
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def safe_name(value):
    p = Path(value)
    if p.is_absolute() or p.drive or '..' in p.parts or '\\' in value:
        raise ValueError('Unsafe archive member: '+value)
    return value


def bind(expected, path, digest):
    key = canonical(path)
    if key in expected and expected[key] != digest:
        raise ValueError('Conflicting registered identity: '+str(path))
    expected[key] = digest


def source_closure(starts):
    """AST-only local import closure, never import the scientific modules."""
    pending, found = list(starts), {}
    while pending:
        path = Path(pending.pop()).resolve()
        if path.name in found:
            if found[path.name] != path:
                raise ValueError('Local source basename collision.')
            continue
        found[path.name] = path
        for n in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names = ([a.name.split('.')[0] for a in n.names] if isinstance(n, ast.Import)
                     else [n.module.split('.')[0]] if isinstance(n, ast.ImportFrom) and n.module else [])
            pending.extend(ROOT/(name+'.py') for name in names if (ROOT/(name+'.py')).is_file())
    return found


def files_under(folder):
    """Only ordinary files; no redirection outside the selected evidence tree."""
    for directory, directories, files in os.walk(folder, followlinks=False):
        for name in [*directories, *files]:
            p = Path(directory)/name
            if p.is_symlink() or getattr(p.stat(follow_symlinks=False), 'st_file_attributes', 0) & 0x400:
                raise ValueError('Archive does not follow symlink/reparse entries: '+str(p))
        for name in files:
            yield Path(directory)/name


def gate(run, audit, log, *, audit_exit_code):
    """Metadata/source identity checks, no raw re-audit or model invocation."""
    if audit_exit_code != 0:
        raise ValueError('Supply ROOT-observed actual audit process exit0.')
    if not (log/'RUN.json').is_file() or any(run.rglob('*.lock')) or any(audit.rglob('*.lock')):
        raise ValueError('Use the real log and inactive whole evidence trees.')
    plan, score = read(run/'REGISTERED.json'), read(run/'SCORED.json')
    ap, passed = read(audit/'REGISTERED.json'), read(audit/'INDEPENDENT_AUDIT.json')
    if (plan.get('namespace') != NAMESPACE or len(plan.get('sources', {})) != 42
            or (plan.get('training_mothers'), plan.get('development_mothers'), plan.get('queries')) != (64, 4, 24)
            or tuple(plan.get('arms', ())) != ARMS or score.get('functional_cycle_complete') is not True
            or score.get('new_held_mothers') != 24 or passed.get('passed') is not True
            or passed.get('coverage_complete') is not True or passed.get('queries') != 24
            or canonical(passed['run']) != canonical(run) or canonical(ap['run']) != canonical(run)
            or passed['audit_registration_sha256'] != sha(audit/'REGISTERED.json')
            or passed['source_sha256'] != sha(ROOT/'audit_aam_formation_local_cycle.py')):
        raise ValueError('Exact complete V5 and actual whole-raw audit PASS required.')
    expected = {}
    for manifest in (plan, ap):
        for category in ('sources', 'inputs', 'expected'):
            for p, digest in manifest.get(category, {}).items():
                bind(expected, p, digest)
    for p in (run/'REGISTERED.json', run/'SCORED.json'):
        if expected.get(canonical(p)) != sha(p):
            raise ValueError('Passed audit does not bind this actual scored run.')
    audit_source = ROOT/'audit_aam_formation_local_cycle.py'
    if ap['sources'].get(str(audit_source.resolve())) != passed['source_sha256']:
        raise ValueError('Audit source is not in its own actual source registration.')
    fp = run/'ALL_PREDICTIONS_AND_SELECTIONS_FROZEN.json'
    if expected.get(canonical(fp)) != sha(fp):
        raise ValueError('Actual prediction freeze differs from the passed audit input.')
    freeze = read(fp)
    if (freeze.get('schema') != 'aam-formation-local-initial-cycle-v5-freeze-A'
            or freeze.get('queries') != 24 or freeze.get('prediction_and_selection_frozen') is not True
            or freeze.get('held_futures_revealed') is not False
            or freeze.get('raw_verification_contract') != 'A+final_full_raw_audit'
            or freeze.get('root_registration_sha256') != sha(run/'REGISTERED.json')
            or freeze.get('sources') != plan['sources']
            or freeze.get('fit_completed_sha256') != sha(run/'FIT_COMPLETED.json')
            or score['prediction_freeze'] != {'path': str(fp), 'sha256': sha(fp)}):
        raise ValueError('Whole prediction/selection and new fitted-model identity required.')
    for p, digest in freeze['all_forecast_files'].items():
        bind(expected, p, digest)
    for p, digest in {**plan['sources'], **ap['sources']}.items():
        if sha(p) != digest:
            raise ValueError('Frozen science/auditor source changed: '+p)
    for group, count, complete in (('train', 64, 'TRAIN_COMPLETE.json'), ('dev', 4, 'QUERY_COMPLETE.json'), ('held', 24, 'EVAL_COMPLETE.json')):
        for mother in range(count):
            path = run/group/f'parent_{mother:03d}'/complete
            if expected.get(canonical(path)) != sha(path):
                raise ValueError('A whole registered data mother differs from its passed identity.')
            value = read(path)
            if value.get('status') != 'complete' or value['binding']['role'] != group or value['binding']['mother_id'] != mother:
                raise ValueError('No missing or replaced train/dev/held mother may be archived.')
    return plan, score, ap, passed, expected


def recovery_resources(run, audit, recovery, recovery_audit):
    """Bind actual recovery provenance and the supplementary audit to this run."""
    if recovery is None and recovery_audit is None:
        if (run/'predictions/query_000/RECOVERY_CONTRACT.json').exists():
            raise ValueError('A recovered run also requires its actual supplementary audit and recovery resources.')
        return {}, {}, [], {}
    if recovery is None or recovery_audit is None:
        raise ValueError('Supply both the recovery directory and its supplementary audit.')
    recovery = Path(recovery).resolve()
    recovery_audit = Path(recovery_audit).resolve()
    registration = read(recovery/'REGISTERED.json')
    supplement = read(recovery_audit/'RECOVERY_AUDIT.json')
    if (supplement.get('passed') is not True or canonical(supplement['run']) != canonical(run)
            or canonical(registration['run']) != canonical(run)
            or supplement['audit_registration_sha256'] != sha(recovery_audit/'REGISTERED.json')
            or supplement['core_audit_sha256'] != sha(audit/'INDEPENDENT_AUDIT.json')
            or supplement['recovery_registration_sha256'] != sha(recovery/'REGISTERED.json')
            or supplement['prediction_freeze_sha256'] != sha(run/'ALL_PREDICTIONS_AND_SELECTIONS_FROZEN.json')):
        raise ValueError('Require the exact actual recovery audit PASS.')
    collection = {}
    expected = read(recovery_audit/'REGISTERED.json')['inputs']
    for name, directory in (('recovery', recovery), ('recovery_audit', recovery_audit)):
        for path in files_under(directory):
            collection[name+'/'+path.relative_to(directory).as_posix()] = path
    inventory_receipt = Path(registration['inventory_receipt']['path']).resolve()
    inventory = read(inventory_receipt)
    for path in files_under(inventory_receipt.parent):
        collection['first_recovery_inventory/'+path.relative_to(inventory_receipt.parent).as_posix()] = path
    for item in inventory['original_mutable_snapshots']+inventory['interrupted_fragment_snapshots']:
        expected[item['snapshot']] = item['sha256']
    sources = {**registration['recovery_sources'], inventory['source']: inventory['source_sha256'],
               str((ROOT/'audit_aam_formation_local_recovery.py').resolve()): supplement['source_sha256']}
    expected.update(sources)
    work = Path(registration['work_review']['path'])
    collection['recovery_review/'+work.name] = work
    contract = read(registration['contract']['path'])
    for origin, digest in contract['source_preflight_proofs'].items():
        path = Path(origin)
        collection['recovery_review/'+path.name] = path
        expected[str(path)] = digest
    return collection, expected, list(map(Path, sources)), {
        'recovery_origin': str(recovery), 'recovery_audit_origin': str(recovery_audit),
        'first_inventory_origin': str(inventory_receipt.parent),
        'supplementary_audit_sha256': sha(recovery_audit/'RECOVERY_AUDIT.json')}


def figure_resources(run, expected):
    receipt = run/'figures'/'V5_PLOT_RECEIPT.json'
    if not receipt.exists():
        return [], []
    plot = read(receipt)
    source = ROOT/'plot_aam_formation_local_v5.py'
    if plot.get('score_sha256') != expected[canonical(run/'SCORED.json')]:
        raise ValueError('Figure receipt is not for the scored V5 results.')
    paths = [receipt]
    for original, digest in plot['figures'].items():
        path = Path(original).resolve()
        if path.parent != receipt.parent:
            raise ValueError('Figures must be under the V5 figures directory.')
        bind(expected, path, digest)
        paths.append(path)
    bind(expected, source, plot['source_sha256'])
    return paths, [source]


def diagnostic_outputs(registration, expected):
    """Verify the terminal receipt binds its registration and list its outputs."""
    completed = registration.parent/'COMPLETED.json'
    if not completed.exists():
        failed = registration.parent/'FAILED.json'
        if not failed.is_file():
            raise ValueError('Post-main diagnostic has no terminal receipt; wait for it to close.')
        return [registration, failed]
    result = read(completed)
    if result.get('complete') is not True or result['input_registration_sha256'] != sha(registration):
        raise ValueError('Post-main diagnostic has no bound completion receipt.')
    paths = [registration, completed]
    for item in result['files'].values():
        path = Path(item['path']).resolve()
        if path.parent != registration.parent or path.stat().st_size != item['bytes']:
            raise ValueError('Diagnostic output path or size does not match its receipt.')
        bind(expected, path, item['sha256'])
        paths.append(path)
    bind(expected, registration, result['input_registration_sha256'])
    return paths


def collect(run, audit, log, *, audit_exit_code, docs=None, recovery=None, recovery_audit=None):
    """Build a complete origin/member map; never recursively load old archives."""
    plan, score, ap, passed, expected = gate(run, audit, log, audit_exit_code=audit_exit_code)
    payload, generated, byorigin = {}, {}, {}
    def include(name, path):
        name, path = safe_name(name), Path(path).resolve()
        if name in payload or name in generated:
            raise ValueError('Duplicate archive member: '+name)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix in ('.log', '.jsonl'):
            content = path.read_bytes()
            generated[name] = {'bytes': content, 'origin': str(path), 'scope': 'Actual mutable-log snapshot', **stamp()}
        else:
            payload[name] = path
        byorigin.setdefault(canonical(path), name)
    for path in sorted(files_under(run)):
        include('experiment/'+path.relative_to(run).as_posix(), path)
    for path in sorted(files_under(audit)):
        include('independent_audit/'+path.relative_to(audit).as_posix(), path)
    additions, recovery_expected, recovery_sources, recovery_map = recovery_resources(run, audit, recovery, recovery_audit)
    for name, path in additions.items():
        include(name, path)
    for path, digest in recovery_expected.items():
        bind(expected, path, digest)
    extra_sources = [ROOT/'audit_aam_formation_local_cycle.py', Path(__file__).resolve(),
                     ROOT/'package_aam_formation_local_capsule.py', *recovery_sources]
    _, figure_sources = figure_resources(run, expected)
    extra_sources.extend(figure_sources)
    diagnostic_registrations = []
    for path in sorted((run/'diagnostics').glob('*/REGISTERED.json')):
        d = read(path)
        diagnostic_outputs(path, expected)
        diagnostic_registrations.append(str(path))
        for category in ('sources', 'inputs'):
            for p, digest in d.get(category, {}).items():
                bind(expected, p, digest)
        extra_sources.extend(Path(p) for p in d.get('sources', {}))
    sources = source_closure([*map(Path, plan['sources']), *map(Path, ap['sources']), *extra_sources])
    for name, path in sorted(sources.items()):
        include('sources/'+name, path)
    documents = {key: Path(path) for key, path in (docs or {}).items()}
    for kind in ('READ_FIRST_ZH', 'RESULTS_ZH', 'ALGORITHM_ZH'):
        documents.setdefault(kind, ROOT/(DOC_PREFIX+kind+'.md'))
    documents['PROTOCOL_ZH'] = Path(plan['protocol'])
    for kind, path in documents.items():
        include('docs/'+DOC_PREFIX+kind+'.md', path)
    include('READ_FIRST_ZH.md', documents['READ_FIRST_ZH'])
    # These exact files, not historical raw/forest/parent ZIPs, are the actual
    # legacy prerequisites that this V5 code and its audit read/hash.
    external = {}
    for manifest in (plan, ap):
        for p, digest in manifest.get('inputs', {}).items():
            path = Path(p).resolve()
            if path.is_relative_to(run) or path.is_relative_to(audit) or canonical(path) in byorigin:
                continue
            bind(external, path, digest)
    for dpath in diagnostic_registrations:
        for p, digest in read(dpath).get('inputs', {}).items():
            path = Path(p).resolve()
            if not path.is_relative_to(run) and not path.is_relative_to(audit) and canonical(path) not in byorigin:
                bind(external, path, digest)
    input_map = []
    for index, (key, digest) in enumerate(sorted(external.items())):
        path = Path(key)
        if sha(path) != digest:
            raise ValueError('Exact external prerequisite changed: '+str(path))
        include(f'prerequisites/{index:03d}_'+path.name, path)
    # Every registered source/input has a concrete embedded origin, even when
    # an input was already included as protocol/source rather than duplicated.
    required = {**plan['sources'], **plan['inputs'], **ap['sources'], **ap['inputs'], **recovery_expected}
    for original, digest in required.items():
        key = canonical(original)
        if key not in byorigin:
            raise ValueError('Required actual dependency has no embedded payload: '+original)
        input_map.append({'original_absolute_path': original, 'member': byorigin[key], 'sha256': digest})
    for path in sorted(log.iterdir()):
        allowed = (path.name in ('RUN.json', 'EVENTS.jsonl', 'CURRENT_CHECKPOINT.json', 'SUMMARY.md',
                                 'WORK_V4_TECHNICAL_PRE_REVIEW_FINAL_ONLY.txt', 'V5_RESOURCE_DEPENDENCY_ANALYSIS_ZH.md')
                   or path.name in REVIEW_FILES
                   or path.name.startswith('V5_') and path.suffix.lower() in ('.json', '.md'))
        if path.is_file() and allowed and canonical(path) not in byorigin:
            include('research_log_snapshot/'+path.name, path)
    history = []
    for name in ('FORMATION_PATH_CYCLE_V4_ARCHIVE_RECEIPT.json', 'FORMATION_AUXILIARY_CYCLE_V3_ARCHIVE_RECEIPT.json'):
        p = log/name
        if not p.is_file():
            p = DEFAULT_LOG/name
        if p.is_file():
            value = read(p)
            if value.get('all_payload_sha_and_CRC_verified') is not True:
                raise ValueError('History archive receipt is not actually complete.')
            include('historical_references/'+name, p)
            history.append({'receipt_member': 'historical_references/'+name,
                            'archive': value['archive'], 'sha256': value['sha256'], 'bytes': value['bytes'],
                            'history_only_not_current_runtime_dependency': True, 'archive_payload_embedded': False})
    restore = {'current_run_origin': str(run), 'experiment_payload': 'experiment/',
        'audit_origin': str(audit), 'audit_payload': 'independent_audit/',
        'recovery': recovery_map,
        'actual_required_dependency_map': input_map, 'historical_archives_reference_only': history,
        'runtime_environment_bundled': False, 'automatic_arbitrary_path_portability_tested': False,
        'instructions': ['Extract to a short path on a volume with sufficient space.',
            'Restore experiment/,independent_audit/,sources/ and each prerequisite to its exact original absolute path using this map,or independently register and audit an explicit relocation procedure.',
            'Frozen registrations/sources are not rewritten by packaging;no one-click arbitrary-location execution is claimed.',
            'This version contains its own64newtraining histories,psi,all5forestfits andmodel bytes;old raw histories,old joblib and giant historical ZIPs are not required for current-version scientific replay/audit.',
            'Exact old macro/micro metric files and registration/PASS prerequisites are embedded,not path-only references.',
            'Install the separately recorded Python package environment. Read saved evidence first;do not silently rerun consumed cohorts or interrupted attempts.']}
    versions = {}
    for name in ('numpy', 'scipy', 'numba', 'scikit-learn', 'threadpoolctl', 'joblib', 'matplotlib'):
        versions[name] = importlib.metadata.version(name)
    if recovery_map:
        versions['psutil'] = importlib.metadata.version('psutil')
    additions = {'RESTORE_PATHS.json': restore,
        'ENVIRONMENT.json': {**stamp(), 'python': sys.version, 'platform': platform.platform(), 'packages': versions,
            'scope': 'Observed packaging environment;not cross-machine runtime validation.'},
        'SOURCE_INPUT_DECLARATION.json': {'sources42': plan['sources'], 'original_inputs': plan['inputs'],
            'whole_audit_sources': ap['sources'], 'recovery': recovery_map, 'required_embedded_dependencies': input_map,
            'post_main_diagnostic_registrations': diagnostic_registrations,
            'post_main_diagnostics_automatically_assumed_audited': False}}
    for name, value in additions.items():
        generated[name] = {'bytes': json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8'),
                           'origin': 'Generated actual packaging declaration'}
    generated['requirements.txt'] = {'bytes': ('# Packaging-time environment\n'+'\n'.join(k+'=='+v for k, v in versions.items())+'\n').encode(), 'origin': 'Observed package versions'}
    return payload, generated, expected, plan, score, passed


def write_zip(output, payload, generated, expected, *, schema, details):
    """Hash actual bytes while copying, then read every member toEOF/SHA/CRC."""
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(output, 'x', allowZip64=True) as archive:
        for name, path in sorted(payload.items()):
            before = path.stat(); digest, count = hashlib.sha256(), 0
            kind = zipfile.ZIP_STORED if path.suffix.lower() in ('.npz', '.png', '.joblib', '.zip') else zipfile.ZIP_DEFLATED
            zi = zipfile.ZipInfo(name); zi.compress_type = kind
            zi._compresslevel = 3
            with path.open('rb') as source, archive.open(zi, 'w', force_zip64=True) as target:
                for block in iter(lambda: source.read(8*1024*1024), b''):
                    target.write(block); digest.update(block); count += len(block)
            actual = digest.hexdigest(); frozen = expected.get(canonical(path)); after = path.stat()
            if (count != before.st_size or before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns
                    or frozen is not None and actual != frozen):
                raise ValueError('Input changed/frozen identity mismatched;partial ZIP retained: '+str(path))
            records.append({'path': name, 'origin': str(path), 'bytes': count, 'sha256': actual,
                            'matched_registered_or_passed_identity': frozen is not None})
        for name, item in sorted(generated.items()):
            content = item['bytes']; archive.writestr(name, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=3)
            records.append({'path': name, 'origin': item['origin'], 'bytes': len(content),
                            'sha256': hashlib.sha256(content).hexdigest()})
        manifest = {**stamp(), 'schema': schema, 'payloads': records, **details}
        archive.writestr('MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2), compress_type=zipfile.ZIP_DEFLATED, compresslevel=3)
    with zipfile.ZipFile(output) as archive:
        names = {r['path'] for r in records} | {'MANIFEST.json'}
        if set(archive.namelist()) != names or len(archive.namelist()) != len(names):
            raise ValueError('ZIP missing/duplicate/orphan members;partial archive retained.')
        if json.loads(archive.read('MANIFEST.json'))['payloads'] != records:
            raise ValueError('Manifest readback mismatch.')
        for record in records:
            digest, count = hashlib.sha256(), 0
            with archive.open(record['path']) as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b''):
                    digest.update(block); count += len(block)
            if digest.hexdigest() != record['sha256'] or count != record['bytes']:
                raise ValueError('Actual all-member SHA/length/CRC readback failed.')
    return {'archive': str(output), 'bytes': output.stat().st_size, 'sha256': sha(output),
            'payload_files': len(records), 'all_payload_sha_and_CRC_verified': True,
            'CRC_method': 'Every member read toEOF,including manifest;ZipExtFile CRC plus actual payloadSHA.',
            'compression': 'Stored already-compressed NPZ/PNG/joblib;DEFLATE3 text. No parent archive payload.'}


def package(run, audit, output, *, log=DEFAULT_LOG, audit_exit_code, docs=None, recovery=None, recovery_audit=None):
    run, audit, output, log = map(lambda p: Path(p).resolve(), (run, audit, output, log))
    receipt = log/'FORMATION_LOCAL_CYCLE_V5_ARCHIVE_RECEIPT.json'
    if output.exists() or receipt.exists() or output.drive.casefold() != 'h:' or output.is_relative_to(run) or output.is_relative_to(audit):
        raise ValueError('Use a new H: ZIP outside both input trees;never overwrite partial output.')
    payload, generated, expected, plan, score, passed = collect(run, audit, log, audit_exit_code=audit_exit_code,
        docs=docs, recovery=recovery, recovery_audit=recovery_audit)
    result = write_zip(output, payload, generated, expected, schema='complete-aam-mu5-local-history-v5-resources',
        details={'full_current_raw_and_code': True, 'exact_required_external_inputs_embedded': True,
                 'historical_parent_archives_embedded': 0, 'runtime_environment_bundled': False,
                 'sources42': len(plan['sources']), 'actual_whole_audit_passed': True,
                 'actual_recovery_audit_passed': True if recovery is not None else None,
                 'scientific_goal_achieved': score.get('scientific_goal_achieved'),
                 'main_audit_scope': passed.get('coverage'), 'no_new_native_fit_score_RNG': True})
    result.update(**stamp(), run=str(run), actual_whole_audit_passed=True,
                  audit_exit_code_observed_by_ROOT=audit_exit_code,
                  actual_recovery_audit_passed=True if recovery is not None else None,
                  full_current_raw_and_code=True, exact_required_external_inputs_embedded=True,
                  historical_parent_archives_embedded=0, new_native_fit_score_RNG=0)
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'audit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--log', type=Path, default=DEFAULT_LOG)
    parser.add_argument('--audit-exit-code', type=int, required=True)
    parser.add_argument('--recovery', type=Path)
    parser.add_argument('--recovery-audit', type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.run, args.audit, args.output, log=args.log,
                             audit_exit_code=args.audit_exit_code, recovery=args.recovery,
                             recovery_audit=args.recovery_audit), ensure_ascii=False))
