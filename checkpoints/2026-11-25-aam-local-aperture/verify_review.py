"""Recompute saved numerical checks without generating or refitting data."""

import argparse
from importlib import import_module
from pathlib import Path

from verify_manifest import verify


CHECKS = (
    ("snapshot", "independent_transition_audit", "audit", ()),
    ("fixed", "independent_fixed_event_audit", "audit", ()),
    ("mobile", "independent_mobile_audit", "audit", ()),
    ("fixed-c", "independent_assay_forecast_audit", "audit", ("EVENT",)),
    ("mobile-c", "independent_assay_forecast_audit", "audit", ("MOBILE",)),
    ("replication-witness", "independent_replication_audit", "witness_audit", ()),
    ("replication-c", "independent_replication_audit", "C_audit", ()),
    ("continuum-witness", "independent_continuum_witness_audit", "calculate", ()),
    ("continuum-c", "independent_continuum_C_audit", "audit", ()),
    ("pulse", "independent_pulse_audit", "audit", ()),
    ("minimal", "independent_minimal_oracle_audit", "minimal", ()),
    ("oracle", "independent_minimal_oracle_audit", "exact_oracle", ()),
    ("proxy", "independent_proxy_audit", "proxy", ()),
    ("calibrated-proxy", "independent_proxy_audit", "calibrated", ()),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checks", nargs="*", metavar="CHECK",
                        help="Checks to run. Empty input runs all 14 checks.")
    args = parser.parse_args()
    available = {name for name, *_ in CHECKS}
    if bad := set(args.checks) - available:
        parser.error(f"Unknown check {', '.join(sorted(bad))}. Available: {', '.join(sorted(available))}.")
    verify(Path(__file__).resolve().parent)
    for name, module, function, params in CHECKS:
        if args.checks and name not in args.checks:
            continue
        getattr(import_module(module), function)(*params)
        print(f"PASS: {name}", flush=True)


if __name__ == "__main__":
    main()
