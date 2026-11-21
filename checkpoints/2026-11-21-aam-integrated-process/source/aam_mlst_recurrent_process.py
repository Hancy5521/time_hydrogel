"""Causal recurrent reconstruction of the finite AAm structural code and history."""

from contextlib import contextmanager
from operator import index

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SCHEMA = 'aam-microstructure-history-gru40-v1'
AUGMENTATIONS = 4
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = .003
WEIGHT_DECAY = 1.e-4
CAMERA_SHAPE = (3,4)
LOCAL_SHAPE = (3,2,3)
CELLS = 6


class _ProcessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(4,24,device='cpu',dtype=torch.float32)
        self.history = nn.GRU(24,40,batch_first=True,device='cpu',dtype=torch.float32)
        self.current = nn.Linear(40,6,device='cpu',dtype=torch.float32)
        self.next_increment = nn.Linear(40,6,device='cpu',dtype=torch.float32)
        self.code = nn.Linear(40,CELLS,device='cpu',dtype=torch.float32)

    def forward(self, cameras):
        hidden, _ = self.history(torch.tanh(self.embedding(cameras)))
        current = self.current(hidden)
        # The increment is a one-step forecast from the earlier hidden state.
        transitions = self.next_increment(hidden[:,:-1])
        return current, transitions, self.code(hidden[:,-1])


@contextmanager
def _deterministic_cpu():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic,warn_only=warn_only)
        torch.set_num_threads(threads)


def _cameras(value):
    x = np.asarray(value,dtype=float)
    if x.shape == CAMERA_SHAPE:
        x = x[None,...]
    if x.ndim != 3 or x.shape[1:] != CAMERA_SHAPE or not len(x) or not np.isfinite(x).all():
        raise ValueError('Use finite three-stage, four-channel camera rows.')
    return x


def _center_scale(value):
    center = value.mean(axis=(0,1))
    scale = value.std(axis=(0,1))
    return center,np.where(scale > 0.,scale,1.)


def _float_tensor(value):
    a = np.array(value,dtype=np.float32,order='C',copy=True)
    if not np.isfinite(a).all():
        raise ValueError('Training and prediction inputs must fit finite CPU float32.')
    return torch.from_numpy(a)


def fit_recurrent(clean,local,labels,noise,seed,*,auxiliary=1.):
    """Fit once on clean training cameras, exact latent stages and six code cells.

    Native Gaussian noise replicas are training inputs for the same mother.
    The no-increment ablation uses auxiliary=0 with the same architecture and
    seeds. It retains reconstruction at each stage and terminal code CE.
    """
    x = _cameras(clean)
    stages = np.asarray(local,dtype=float)
    target = np.asarray(labels)
    sigma = np.broadcast_to(np.asarray(noise,dtype=float),CAMERA_SHAPE)
    seed = index(seed)
    coefficient = float(auxiliary)
    if (stages.shape != (len(x),*LOCAL_SHAPE) or not np.isfinite(stages).all()
            or target.shape != (len(x),) or target.dtype.kind not in 'iu'
            or not np.array_equal(np.unique(target),np.arange(CELLS))
            or not np.isfinite(sigma).all() or np.any(sigma < 0.)
            or seed < 0 or not np.isfinite(coefficient) or coefficient < 0.):
        raise ValueError('Use finite aligned clean training history, six occupied code cells and valid noise/seed.')

    camera_mean = x.mean(axis=0)
    camera_scale = np.hypot(x.std(axis=0),sigma)
    camera_scale = np.where(camera_scale > 0.,camera_scale,1.)
    stage_flat = stages.reshape(len(x),3,6)
    delta = np.diff(stage_flat,axis=1)
    stage_mean,stage_scale = _center_scale(stage_flat)
    delta_mean,delta_scale = _center_scale(delta)
    aug_seed,shuffle_seed,model_seed = np.random.SeedSequence(seed).spawn(3)
    perturb = np.random.default_rng(aug_seed).standard_normal((len(x),AUGMENTATIONS,*CAMERA_SHAPE))
    normalized = (x[:,None]+sigma*perturb-camera_mean)/camera_scale
    samples = _float_tensor(normalized.reshape(-1,*CAMERA_SHAPE))
    train_stage = _float_tensor(np.repeat((stage_flat-stage_mean)/stage_scale,AUGMENTATIONS,axis=0))
    train_delta = _float_tensor(np.repeat((delta-delta_mean)/delta_scale,AUGMENTATIONS,axis=0))
    cells = torch.from_numpy(np.repeat(target.astype(np.int64),AUGMENTATIONS))
    shuffle = np.random.default_rng(shuffle_seed)
    trace = np.empty((EPOCHS,3))

    with _deterministic_cpu():
        torch.manual_seed(int(model_seed.generate_state(1,dtype=np.uint64)[0]) & ((1<<63)-1))
        net = _ProcessNet()
        optimizer = torch.optim.AdamW(net.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
        for epoch in range(EPOCHS):
            order = shuffle.permutation(len(samples))
            totals = np.zeros(3)
            for start in range(0,len(order),BATCH_SIZE):
                ids = order[start:start+BATCH_SIZE]
                now,step,logits = net(samples[ids])
                ce = F.cross_entropy(logits,cells[ids])
                absolute = F.mse_loss(now,train_stage[ids])
                increments = F.mse_loss(step,train_delta[ids])
                optimizer.zero_grad(set_to_none=True)
                (ce+absolute+coefficient*increments).backward()
                optimizer.step()
                totals += len(ids)*np.array([ce.item(),absolute.item(),increments.item()])
            trace[epoch] = totals/len(samples)
        state = {name:value.detach().numpy().copy() for name,value in net.state_dict().items()}
    if not np.isfinite(trace).all() or any(not np.isfinite(parameter).all() for parameter in state.values()):
        raise ArithmeticError('Recurrent training did not produce a finite model.')
    return {'schema':SCHEMA,'classes':np.arange(CELLS),'weights':state,'seed':seed,'auxiliary':coefficient,
            'train_mothers':len(x),'augmentations':AUGMENTATIONS,'epochs':EPOCHS,'batch_size':BATCH_SIZE,
            'learning_rate':LEARNING_RATE,'weight_decay':WEIGHT_DECAY,'camera_mean':camera_mean,
            'camera_scale':camera_scale,'stage_mean':stage_mean,'stage_scale':stage_scale,
            'delta_mean':delta_mean,'delta_scale':delta_scale,'loss_columns':('code_ce','current_mse','next_increment_mse'),
            'training_losses':trace}


def predict_recurrent(model,query):
    """Return readouts for each seen prefix, next increments, and final code mass."""
    if model['schema'] != SCHEMA:
        raise ValueError('Load the original recurrent structural model schema.')
    x = _cameras(query)
    with _deterministic_cpu(),torch.inference_mode():
        net = _ProcessNet()
        net.load_state_dict({name:_float_tensor(value) for name,value in model['weights'].items()})
        xx = _float_tensor((x-model['camera_mean'])/model['camera_scale'])
        now,step,logits = net(xx)
        probabilities = torch.softmax(logits.to(dtype=torch.float64),dim=1).numpy().copy()
        current = now.numpy()*model['stage_scale']+model['stage_mean']
        transitions = step.numpy()*model['delta_scale']+model['delta_mean']
    if not all(np.isfinite(value).all() for value in (probabilities,current,transitions)):
        raise ArithmeticError('Recurrent inference requires finite probabilities and readouts.')
    return {'probabilities':probabilities,'stage_values':current.reshape(len(x),*LOCAL_SHAPE),
            'transition_values':transitions.reshape(len(x),2,2,3)}


def synthetic_fixture():
    rng = np.random.default_rng(2712)
    label = np.repeat(np.arange(CELLS),4)
    x = rng.normal(0.,.2,size=(len(label),*CAMERA_SHAPE))
    x[:,:,0] += label[:,None]/3.
    states = np.concatenate((x,np.cumsum(x[:,:,:2],axis=1)),axis=-1).reshape(len(x),*LOCAL_SHAPE)
    model = fit_recurrent(x,states,label,np.full(4,.02),27213)
    pair = np.repeat(x[:1],2,axis=0)
    reference = predict_recurrent(model,pair)
    pair[1,2] += 2.
    result = predict_recurrent(model,pair)
    np.testing.assert_allclose(result['probabilities'].sum(axis=1),1.,rtol=0.,atol=5.e-15)
    np.testing.assert_array_equal(reference['stage_values'][:,:2],result['stage_values'][:,:2])
    np.testing.assert_array_equal(reference['transition_values'],result['transition_values'])
    if np.allclose(result['stage_values'][0,2],result['stage_values'][1,2]):
        raise AssertionError('The synthetic current-stage head ignored the current camera.')
    repeat = fit_recurrent(x,states,label,np.full(4,.02),27213)
    for name,weight in repeat['weights'].items():
        np.testing.assert_array_equal(weight,model['weights'][name])
    return {'donors':len(x),'bitwise_deterministic':True,'no_future_frame_dependence':True,
            'first_epoch_losses':model['training_losses'][0].tolist(),
            'last_epoch_losses':model['training_losses'][-1].tolist()}


if __name__ == '__main__':
    print(synthetic_fixture())
