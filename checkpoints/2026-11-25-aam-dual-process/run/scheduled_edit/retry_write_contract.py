"""Output-only retry for a positional argument collision in the frozen driver."""

from argparse import ArgumentParser
import numpy as np
import scheduled


def output_archive(destination, /, **fields):
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **fields)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--freeze-sha256", required=True)
    options = parser.parse_args()
    scheduled.arrays = output_archive
    scheduled.confirm(options.workers, options.freeze_sha256)
