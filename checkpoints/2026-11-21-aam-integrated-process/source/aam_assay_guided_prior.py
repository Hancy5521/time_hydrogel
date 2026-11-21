"""Cheap, full-support initial-history proposals under the original mu5 prior."""
import operator

import numpy as np

from aam_formation_local_initial import clone_initial, draw_initial, initial_fingerprint, validate_initial


PORTS = (1.45,4.45)
NAMES = ('initiators','MBAA_left', 'AAm_aperture_left','AAm_aperture_right',
         'MBAA_aperture_left','MBAA_aperture_right','MBAA_x_left','MBAA_x_right')
FLOOR = .125
RIDGE = 4.
TIME_FACTOR = 3.
MAX_CHEAP_DRAWS = 32
GRID = 1 << 53


def signature(initial):
    """Eight pre-reaction statistics, excluding the delayed six molecules."""
    xy = np.asarray(initial['points'],float)[:72,:2]
    if xy.shape != (72,2) or not np.isfinite(xy).all():
        raise ValueError('The explicit initial proposal needs the fixed72present particles.')
    dx = (xy[:,0,None]-np.asarray(PORTS)+3.) % 6. - 3.
    dy = (xy[:,1]-1.35+3.) % 6. - 3.
    x = np.maximum(0.,1.-np.abs(dx)/1.3)
    w = x*np.maximum(0.,1.-np.abs(dy)/1.3)[:,None]
    return np.r_[float(initial['I']), np.mean(initial['left']), w[:64].sum(axis=0),
                 w[64:].sum(axis=0), x[64:].sum(axis=0)]


def _ridge(x,y):
    center = x.mean(axis=0)
    s = x.std(axis=0)
    scale = np.where(s > 0.,s,1.)
    xz = (x-center)/scale
    mean = y.mean(axis=0)
    coeff = np.linalg.solve(xz.T@xz + RIDGE*np.eye(x.shape[1]),xz.T@(y-mean))
    return {'center':center,'scale':scale,'mean':mean,'coefficients':coeff}


def predict_clean(fit, descriptors):
    x = np.asarray(descriptors,float)
    if x.ndim < 1 or x.shape[-1] != len(NAMES) or not np.isfinite(x).all():
        raise ValueError('Predict only the fixed finite eight-feature descriptor.')
    return (x-fit['center'])/fit['scale'] @ fit['coefficients'] + fit['mean']


def fit_proxy(descriptors, clean, camera_noise):
    """Fixed ridge mean, honest within-bank LOO errors, known camera noise floor."""
    x,y = np.asarray(descriptors,float), np.asarray(clean,float)
    sigma = np.asarray(camera_noise,float)
    if y.shape == (len(x),3,4):
        y = y.reshape(len(x),12)
    if (x.ndim != 2 or x.shape[1] != len(NAMES) or len(x) < 4 or y.shape != (len(x),12)
            or sigma.shape != (4,) or np.any(sigma <= 0.)
            or not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(sigma).all())):
        raise ValueError('Use only a complete declared training cohort, real clean12 and original noise.')
    loo = np.empty_like(y)
    indexes = np.arange(len(x))
    for i,features in enumerate(x):
        train = indexes != i
        loo[i] = predict_clean(_ridge(x[train],y[train]),features)
    variance = np.mean(np.square(loo-y),axis=0)
    return {**_ridge(x,y),'clean_loo':loo,'residual_variance':variance,
            'proxy_sigma':np.sqrt(np.tile(sigma,3)**2+variance)}


def log_proxy(fit, descriptors, measured):
    y = np.asarray(measured,float)
    if y.shape == (3,4):
        y = y.reshape(12)
    if y.shape != (12,) or not np.isfinite(y).all():
        raise ValueError('A proposal can receive only the original finite noisy 3×4 camera readings.')
    with np.errstate(over='ignore'):
        squared = np.square((predict_clean(fit,descriptors)-y)/fit['proxy_sigma'])
        return -.5*np.sum(squared,axis=-1)/TIME_FACTOR


def proposal_threshold(ell):
    """Exact mass on the implementation's uniform53 cheap-decision grid."""
    ell = np.asarray(ell,float)
    if np.isnan(ell).any() or np.any(ell > 0.):
        raise ValueError('Only the nonpositive camera proxy is a valid guide.')
    return np.ceil(GRID*(FLOOR+(1.-FLOOR)*np.exp(ell))).astype(np.int64)


def guide_mass(fit, descriptors, measured):
    """The guide is ancillary. It never replaces the exact camera likelihood."""
    return proposal_threshold(log_proxy(fit,descriptors,measured))/GRID


def draw_guided_initial(config, measured, fit, rng, *, max_attempts=MAX_CHEAP_DRAWS):
    """One capped block of fresh prior trials. Successful draws follow mu5*a/Z."""
    attempts = operator.index(max_attempts)
    if not 1 <= attempts <= MAX_CHEAP_DRAWS:
        raise ValueError('Use at most the registered32 cheap attempts.')
    draws = []
    for i in range(1,attempts+1):
        new = draw_initial(config,rng)
        threshold = int(proposal_threshold(log_proxy(fit,signature(new),measured)))
        uniform = int(rng.integers(GRID,dtype=np.int64))
        chosen = uniform < threshold
        draws.append({'threshold':threshold,'uniform53':uniform,'chosen':chosen})
        if chosen:
            return new,{'status':'success','attempts':i,'candidate_threshold':threshold,
                        'candidate_mass':threshold/GRID,'cheap_draws':draws}
    return None,{'status':'exhausted','attempts':attempts,'candidate_threshold':None,
                 'candidate_mass':None,'cheap_draws':draws}


def propose_guided_initial(config, old, measured, fit, rng, *, max_attempts=MAX_CHEAP_DRAWS):
    """Return a V5-style initialization proposal, with exact reverse correction.

    Exhaustion has a query-only probability and produces an explicit identity.
    All native interval seeds remain fixed when this proposal is used in replay.
    """
    validate_initial(config,old)
    before = int(proposal_threshold(log_proxy(fit,signature(old),measured)))
    new,metadata = draw_guided_initial(config,measured,fit,rng,max_attempts=max_attempts)
    metadata.update(kind='assay_guided_initial',before_threshold=before,self_loop=True)
    if new is None:
        return clone_initial(old),0.,metadata
    ratio = float(np.log(before/metadata['candidate_threshold']))
    metadata['self_loop'] = initial_fingerprint(new) == initial_fingerprint(old)
    return new,ratio,metadata
