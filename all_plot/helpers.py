import numpy as np


def load_trail(obj_num, bd, category='sample', path='./raw_data/'):

    files = [np.load(f"{path}{obj_num}{bd}obj_trail_{i}.npy") for i in range(1, 6)]
    max_len = max(f.shape[1] for f in files)
    padded = []
    for f in files:
        last = f[:, -1:]
        pad = np.tile(last, (1, max_len - f.shape[1]))
        padded.append(np.concatenate([f, pad], axis=1)[:, 1:])
    arr = np.array(padded)

    if category == 'sample':
        t = arr[:, 0, :].T
    elif category == 'ETE':
        t = arr[:, 1, :].T
    hv = arr[:, 3, :].T
    return t.mean(axis=1), np.mean(hv+1, axis=1), np.min(hv+1, axis=1), np.max(hv+1, axis=1)

