"""Supply the read-only co-observed control missing from the frozen call site."""
import numpy as np
import mobile_fit

source = mobile_fit.ROOT / "MOBILE_CHECK.npz"
with np.load(source, allow_pickle=False) as record:
    mobile_fit.fixed = record["fixed"]
mobile_fit.predict()

