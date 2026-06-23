import warnings
import numpy as np
import torch

def interp_nan_inplace(arr, axis):
    """
    Interpolates NaN values in-place along a specified axis using linear interpolation.

    Parameters
    ----------
    arr : np.ndarray
        A NumPy array potentially containing NaN values. This array is modified in-place.
    axis : int
        The axis along which to perform interpolation.
    """
    if arr.ndim <= axis:
        raise ValueError('Axis is out of bounds for the array')

    # Move the target axis to the front for easier iteration
    arr_swap = np.moveaxis(arr, axis, 0)
    shape = arr_swap.shape

    for idx in np.ndindex(*shape[1:]):
        vec = arr_swap[(slice(None),) + idx]
        nan_mask = np.isnan(vec)
        if nan_mask.any():
            real_mask = ~nan_mask
            xp = np.flatnonzero(real_mask)
            fp = vec[real_mask]
            x = np.flatnonzero(nan_mask)
            vec[nan_mask] = np.interp(x, xp, fp)


def sorted_edit_distance(state1, state2):
    dist = 0
    for h1, h2 in zip(sorted(state1), sorted(state2)):
        if h1 != h2:
            dist += 1
    return dist


def numpy_to_torch(x):
    """
    Convert a NumPy array or masked array to the appropriate Torch object.

    Rules:
      - Plain ndarray -> torch.Tensor
      - np.ma.MaskedArray -> masked_tensor(data, mask)
        with mask polarity corrected for Torch.
    """
    if isinstance(x, np.ma.MaskedArray):
        warnings.filterwarnings("ignore", module="torch.masked.maskedtensor.core")
        return torch.masked.masked_tensor(
            torch.from_numpy(np.asarray(x.data, dtype=x.dtype)),
            torch.from_numpy((~x.mask).astype(bool))
        )

    # plain ndarray
    return torch.from_numpy(np.asarray(x))


def concat_arrays(X):
    if any(isinstance(x, np.ma.MaskedArray) for x in X):
        arr = np.ma.concatenate(X)
        if np.asarray(arr.mask).shape == ():
            arr.mask = np.zeros_like(arr.data, dtype=bool)
        return arr
    return np.concatenate(X)


def data_and_mask(X):
    if isinstance(X, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(X)
        return X.filled(0).astype(float), ~mask
    return np.asarray(X, dtype=float), np.ones_like(X, dtype=bool)


def as_float_array(X):
    if isinstance(X, np.ma.MaskedArray):
        return X.filled(0).astype(float)
    return np.asarray(X, dtype=float)


def torch_to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def mask_array_zeros(X, axis=1):
    # allow negative axes
    if axis < 0:
        axis += X.ndim
    if not (0 <= axis < X.ndim):
        raise ValueError("axis out of range")
    Xm = np.moveaxis(X, axis, 0)
    other_axes = tuple(range(1, X.ndim))
    mask = Xm.sum(axis=other_axes) == 0
    reshape = [1] * X.ndim
    reshape[axis] = -1
    mask = np.broadcast_to(mask.reshape(reshape), X.shape)
    return np.ma.array(X, mask=mask)


def format_params(params, columns, indent=2):
    columns = [str(c) for c in columns]
    rows = []

    for k, v in sorted(params.items()):
        arr = np.asarray(v)

        if arr.ndim == 0:
            values = [f"{float(arr):.4g}"] * len(columns)
        else:
            values = [f"{x:.4g}" for x in arr]

        rows.append([k, *values])

    header = ["param", *columns]
    table = [header, *rows]

    widths = [
        max(len(str(row[i])) for row in table)
        for i in range(len(header))
    ]

    lines = []
    for row in table:
        line = " ".join(
            str(x).rjust(widths[i])
            for i, x in enumerate(row)
        )
        lines.append(" " * indent + line)

    return "\n".join(lines)