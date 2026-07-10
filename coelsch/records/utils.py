import numpy as np


def validate_data(obj, expected_depth, expected_dtype):
    """
    Recursively validates that the object conforms to the expected depth and data type.

    Parameters
    ----------
    obj : dict or any
        The object to validate.
    expected_depth : int
        The expected depth of the object.
    expected_dtype : type
        The expected data type for the values in the object.

    Raises
    ------
    ValueError
        If the object depth or data type does not match the expected values.
    """
    def _validate_recursive(obj, depth):
        if isinstance(obj, dict):
            for key, val in obj.items():
                if not isinstance(key, str):
                    raise ValueError(f'Key {key} at depth {depth} is not str type')
                _validate_recursive(val, depth + 1)
        else:
            if depth != expected_depth:
                raise ValueError(f'Depth of object is not {expected_depth} at all locations')
            if not isinstance(obj, expected_dtype):
                raise ValueError(f'Object value do not match dtype {expected_dtype}: {type(obj)}')

    _validate_recursive(obj, 0)


def quantise(arr, precision):
    levels = 10 ** precision + 1
    q = np.clip((arr * (levels - 1)).round(), 0, levels - 1)
    return q.astype(np.uint16), levels


def dequantise(q, levels, dtype):
    return q.astype(dtype) / (levels - 1)


def run_length_encode(arr):
    if len(arr) == 0:
        return np.array([], dtype=arr.dtype), np.array([], dtype=np.uint32)
    diffs = np.diff(arr)
    change_indices = np.flatnonzero(diffs != 0) + 1
    starts = np.concatenate(([0], change_indices))
    ends = np.concatenate((change_indices, [len(arr)]))
    values = arr[starts]
    lengths = ends - starts
    return values, lengths


def run_length_decode(values, lengths):
    return np.repeat(values, lengths)


def array_encoder_full(arr, precision):
    return  {
        'shape': arr.shape,
        'dtype': arr.dtype.str,
        'data': [round(float(v), precision) for v in arr.ravel()]
    }


def array_encoder_rle(arr, precision):
    quants, levels = quantise(arr.ravel(), precision)
    vals, lens = run_length_encode(quants)
    return {
        'shape': arr.shape,
        'dtype': arr.dtype.str,
        'data': (vals.tolist(), lens.tolist(), levels)
    }


def array_encoder_sparse(arr, precision):
    shape = arr.shape
    arr = arr.ravel()
    idx = np.nonzero(arr)[0]
    val = [round(float(v), precision) for v in arr[idx]]
    return {
        'shape': shape,
        'dtype': arr.dtype.str,
        'data': (idx.tolist(), val)
    }


def array_decoder_full(json_obj):
    return np.array(
        json_obj['data'],
        dtype=json_obj['dtype']
    ).reshape(json_obj['shape'])


def array_decoder_rle(json_obj):
    shape = json_obj['shape']
    dtype = json_obj['dtype']
    vals, lens, levels = json_obj['data']
    quants = run_length_decode(vals, lens)
    arr = dequantise(quants, levels, dtype).reshape(shape)
    return arr


def array_decoder_sparse(json_obj):
    shape = json_obj['shape']
    arr = np.zeros(shape=np.prod(shape), dtype=json_obj['dtype'])
    idx, val = json_obj['data']
    arr[idx] = val
    return arr.reshape(shape)


def _overwrite(obj, key, new_val):
    return new_val


def _add(obj, key, new_val):
    try:
        return obj[key] + new_val
    except KeyError:
        return new_val


def _subtract(obj, key, new_val):
    try:
        return obj[key] - new_val
    except KeyError:
        return -new_val


def _overwrite_ignore_nan(obj, key, new_val):
    try:
        curr_val = obj[key]
    except KeyError:
        return new_val

    if isinstance(new_val, np.ndarray):
        return np.where(np.isnan(new_val), curr_val, new_val)
    elif np.isnan(new_val):
        return curr_val
    else:
        return new_val


def deep_operator(obj, other, method='overwrite'):
    """
    Recursively operates on a dictionary with the values from another dictionary.

    Parameters
    ----------
    obj : dict
        The dictionary to update.
    other : dict
        The dictionary with values to update into `obj`.

    Returns
    -------
    dict
        The updated dictionary.
    """

    if method == 'overwrite':
        operator = _overwrite
    elif method == 'add':
        operator = _add
    elif method == 'subtract':
        operator = _subtract
    elif method == 'overwrite_ignore_nan':
        operator = _overwrite_ignore_nan
    else:
        raise NotImplementedError()

    def _operate_recursive(obj, other):
        for key, val in other.items():
            if isinstance(val, dict):
                obj[key] = _operate_recursive(obj.setdefault(key, {}), val)
            else:
                obj[key] = operator(obj, key, val)
        return obj

    _operate_recursive(obj, other)
