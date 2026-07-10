from copy import copy, deepcopy
import numpy as np

from .utils import (
    validate_data, deep_operator,
    array_encoder_full, array_encoder_rle, array_encoder_sparse,
    array_decoder_full, array_decoder_rle, array_decoder_sparse,
)

DTYPES = {
    'str': str, 'int': int, 'float': float, 'bool': bool,
    'list': list, 'tuple': tuple,
    'set': set, 'frozenset': frozenset,
    'ndarray': np.ndarray,
}


class DeepKeyError(KeyError):
    def __init__(self, missing_key, path):
        super().__init__(missing_key)
        self.missing_key_path = path + (missing_key,)

    def __str__(self):
        kp = ' -> '.join(repr(k) for k in self.missing_key_path)
        return f'Key not found at key path {kp}'


class NestedData:

    """
    A class for storing and manipulating hierarchical data with multi-level dictionaries.

    Attributes
    ----------
    levels : tuple of str
        The levels of nested data hierarchy (e.g., 'group', 'cb', 'chrom', 'other').
    dtype : tuple of type
        The expected data types for the nested data values.
    nlevels : int
        The number of levels in the nested data hierarchy.

    Methods
    -------
    keys()
        Returns top level keys for the nested data object
    values()
        Returns top level values for the nested data object
    get_level_keys(level)
        Returns a sorted list of the unique keys for a particular level
    update(other)
        Updates the nested data dictionary with another `NestedData` or dictionary.
    new_like()
        Returns an empty nested data object with the same level/dtype structure
    filter(whitelist, level=0, inplace=True)
        Filters nested data by keeping only the keys in the whitelist at the specified level.
    """

    def __init__(self, levels, dtype, data=None):
        """
        Initializes the NestedData with the given levels and data types.

        Parameters
        ----------
        levels : tuple of str
            The levels in the nested data hierarchy.
        dtype : tuple of type
            The expected data types for nested data values.
        data : dict, optional
            The initial data. If None, an empty dictionary is used.

        Raises
        ------
        ValueError
            If the levels or data types are invalid.
        """
        if not isinstance(levels, tuple):
            if isinstance(levels, str):
                levels = (levels, )
            else:
                levels = tuple(levels)
        if not levels:
            raise ValueError('Must have at least one level')
        if len(set(levels)) != len(levels):
            raise ValueError('One or more levels are duplicated')
        self.levels = levels
        if not isinstance(dtype, tuple):
            try:
                dtype = tuple(dtype)
            except TypeError:
                dtype = (dtype, )
        for d in dtype:
            if not isinstance(d, type):
                raise ValueError('dtypes should be types')
        self.dtype = dtype
        if data is None:
            self._data = {}
        else:
            validate_data(data, self.nlevels, self.dtype)
            self._data = data

    @property
    def nlevels(self):
        return len(self.levels)

    @classmethod
    def new_like(cls, obj, copy=True):
        if not isinstance(obj, cls):
            raise ValueError(f'obj is not of type {cls.__name__}')
        instance = cls.__new__(cls)
        for attr, val in obj.__dict__.items():
            if attr == '_data':
                instance._data = {}  # create empty data
            else:
                val = deepcopy(val) if copy else val
                setattr(instance, attr, val)
        return instance

    def _resolve_indices(self, index):
        if not isinstance(index, tuple):
            index = (index,)

        if len(index) > self.nlevels:
            raise KeyError(f"too many indices to object with {self.nlevels} levels")

        # Validate types
        for idx in index:
            if isinstance(idx, slice):
                if idx != slice(None):
                    raise KeyError('Only full slices (:) are allowed')
            elif not isinstance(idx, (str, type(Ellipsis), list)):
                raise TypeError(
                    f'Invalid index type {type(idx).__name__}: only str, Ellipsis, '
                    f'list, or full slices (:) are allowed.'
                )

        # Expand ellipsis if present
        if Ellipsis in index:
            idx_ellipsis = index.index(Ellipsis)
            n_missing = self.nlevels - (len(index) - 1)
            index = index[:idx_ellipsis] + (slice(None),) * n_missing + index[idx_ellipsis + 1:]

        return index

    def __getitem__(self, index):
        """
        Advanced indexing into a nested dict

        Parameters
        ----------
        index : str, Ellipsis, slice(None), or tuple of them
            Indexing pattern, allowing partial selection across levels.

        Returns
        -------
        NestedData
            A new NestedData object with reduced levels if applicable.
        """

        cls = type(self)
        index = self._resolve_indices(index)

        # if all indices are str, we simply traverse the nested dicts
        if all(isinstance(idx, str) for idx in index):
            selected_data = self._data
            for i, idx in enumerate(index):
                try:
                    selected_data = selected_data[idx]
                except KeyError:
                    raise DeepKeyError(idx, index[:i])
            if isinstance(selected_data, self.dtype):
                return selected_data
            else:
                selection = cls.new_like(self, copy=False)
                selection.levels = self.levels[len(index):]
                selection._data = selected_data
                return selection

        # otherwise, we select the relevant data from the nested structure and return as a new NestedData object
        selected_data = {}

        def _select(obj, depth, key_path):
            if depth == len(index):
                # End of path, set value
                sd = selected_data
                for key in key_path[:-1]:
                    sd = sd.setdefault(key, {})
                sd[key_path[-1]] = obj
                return None

            idx = index[depth]
            if isinstance(idx, slice):
                for key, val in obj.items():
                    _select(val, depth + 1, key_path + (key,))
            elif isinstance(idx, list):
                for key in idx:
                    if key not in obj:
                        raise DeepKeyError(key, key_path)
                    _select(obj[key], depth + 1, key_path + (key,))
            else:
                # Allow index by key only
                if idx not in obj:
                    raise DeepKeyError(idx, key_path)
                _select(obj[idx], depth + 1, key_path)

        _select(self._data, 0, ())

        # right pad index with None to retain levels below those indexed
        index = index + (slice(None),) * (self.nlevels - len(index))
        # New levels: drop levels where index was a specific value (not a slice or list)
        new_levels = tuple(lvl for lvl, idx in zip(self.levels, index) if isinstance(idx, (slice, list)))
        selection = cls.new_like(self, copy=False)
        selection.levels = new_levels
        selection._data = selected_data
        return selection

    def _setitem_tuple(self, index, value):
        n_idx = len(index)
        if n_idx > self.nlevels:
            raise IndexError('Too many indices provided')
        for lvl in index:
            if not isinstance(lvl, str):
                raise ValueError('nested data keys must be strings')
        validate_data(value, self.nlevels - n_idx, self.dtype)
        subdict = self._data
        for key in index[:-1]:
            if key not in subdict:
                subdict[key] = {}
            subdict = subdict[key]
        subdict[index[-1]] = value

    def __setitem__(self, index, value):
        if isinstance(index, str):
            validate_data(value, self.nlevels - 1, self.dtype)
            self._data[index] = value
        elif index is Ellipsis or index == slice(None):
            validate_data(value, self.nlevels, self.dtype)
            self._data = value
        elif isinstance(index, tuple):
            self._setitem_tuple(index, value)
        else:
            raise KeyError(f'key {index} not supported')

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __delitem__(self, key):
        del self._data[key]

    def __len__(self):
        return len(self._data)

    def __repr__(self):

        def _gather_leaves(d, path=()):
            if not isinstance(d, dict):
                yield path, d
            else:
                for k, v in d.items():
                    yield from _gather_leaves(v, path + (k,))

        preview = {}
        leaves = _gather_leaves(self._data)
        for path, val in leaves:
            key = ' -> '.join(repr(k) for k in path)
            if isinstance(val, np.ndarray):
                val_repr = f"numpy.ndarray({val.shape}, dtype={val.dtype})"
            else:
                val_repr = repr(val)
            preview[key] = val_repr
            if len(preview) == 3:
                break
        try:
            next(leaves)
            preview['...'] = '...'
        except StopIteration:
            pass

        dtype = tuple(d.__name__ for d in self.dtype)
        preview_str = ', '.join(f"{k}: {v}" for k, v in preview.items())

        # Here's the changed part
        preview_lines = [f"        {k}: {v}" for k, v in preview.items()]
        preview_str = ',\n'.join(preview_lines)

        return (f"{self.__class__.__name__}(\n"
                f"    levels={self.levels},\n"
                f"    dtype={dtype},\n"
                f"    data_preview={{\n{preview_str}\n"
                f"    }}\n"
                f")")

    def _validate_other(self, other):
        if isinstance(other, type(self)):
            if self.levels != other.levels:
                raise ValueError('cannot merge nested data, levels do not match')
            if self.dtype != other.dtype:
                raise ValueError('cannot merge nested data, dtypes do not match')
            other = other._data
        if not isinstance(other, dict):
            raise ValueError(f'other must be of type {self.__class__.__name__} or dict')
        validate_data(other, self.nlevels, self.dtype)
        return other

    def update(self, other, update_method='overwrite'):
        """
        Updates the nested data dictionary with another `NestedData` or dictionary.

        Parameters
        ----------
        other : NestedData or dict
            The nested data dictionary or dictionary to update from.
        update_method: str
            The method to use for updating. Options are "overwrite", "overwrite_ignore_nan", 
            or "add" 

        Raises
        ------
        ValueError
            If the nested data structures are incompatible.
        """
        other = self._validate_other(other)
        deep_operator(self._data, other, method=update_method)

    def _op(self, other, method, inplace=False):
        other = self._validate_other(other)
        if inplace:
            s = self
        else:
            s = self.copy()
        deep_operator(s._data, other, method=method)
        return s

    def __add__(self, other):
        return self._op(other, method='add', inplace=False)

    def __iadd__(self, other):
        return self._op(other, method='add', inplace=True)

    def __sub__(self, other):
        return self._op(other, method='subtract', inplace=False)

    def __isub__(self, other):
        return self._op(other, method='subtract', inplace=True)

    def keys(self):
        return self._data.keys()

    def values(self):
        for key in self.keys():
            yield self[key]

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def deep_items(self):
        n = self.nlevels

        def _recursive_items(obj, depth, key_path):
            if depth == n:
                yield key_path, obj
            else:
                for key, val in obj.items():
                    yield from _recursive_items(val, depth + 1, key_path + (key,))

        yield from _recursive_items(self._data, 0, tuple())

    def setdefault(self, index, default):
        try:
            return self[key]
        except KeyError:
            self[key] = default
            return self[key]

    def asdict(self):
        """Return a copy of the raw underlying dictionary data"""
        return copy(self._data)

    def get(self, index, value=None):
        try:
            return self[index]
        except KeyError:
            return value

    def pop(self, key):
        return self._data.pop(key)

    def _get_level_idx(self, level_name):
        if isinstance(level_name, str):
            level_idx = 0
            for lvl in self.levels:
                if lvl == level_name:
                    break
                level_idx += 1
            else:
                raise ValueError(f'level "{level_name}" not recognised')
        elif isinstance(level_name, int):
            if level_name >= self.nlevels:
                raise ValueError('level index is too deep for nested data')
            level_idx = level_name
        else:
            raise ValueError('level must be either a str or int')
        return level_idx

    def get_level_keys(self, level):
        """
        Returns the keys in the nested data object at the specified level.

        Parameters
        ----------
        level : int or str, optional
            The level name or index to filter on.

        Returns
        -------
        list
            The sorted list of unique keys in the nested data object at the specified level
        """
        level = self._get_level_idx(level)
        keys = set()

        def _recursive_search_keys(obj, curr_lvl):
            if curr_lvl == level:
                keys.update(set(obj.keys()))
            else:
                for val in obj.values():
                    _recursive_search_keys(val, curr_lvl + 1)

        _recursive_search_keys(self._data, 0)
        return sorted(keys)

    def copy(self):
        '''
        create a copy of a nested data object
        
        Returns
        -------
        NestedData
            New copied instance
        '''
        new_instance = self.new_like(self)
        new_instance._data = deepcopy(self._data)
        return new_instance

    def add_level_suffix(self, suffix, level=0, inplace=False):
        '''append a suffix to a specified level of a nested data object'''
        level = self._get_level_idx(level)

        if inplace:
            s = self
        else:
            s = self.copy()

        def _recursive_add_suffix(obj, curr_lvl):
            if curr_lvl == level:
                return {f'{key}_{suffix}': val for key, val in obj.items()}
            else:
                for key in list(obj.keys()):
                    obj[key] = _recursive_add_suffix(obj[key], curr_lvl + 1)
            return obj
        s._data = _recursive_add_suffix(s._data, 0)
        if not inplace:
            return s

    def filter(self, whitelist, level=0, inplace=True, with_copy=False):
        """
        Filters the nested data dictionary by retaining only keys in the whitelist at the specified level.

        Parameters
        ----------
        whitelist : set of str
            The set of keys to retain at the specified level.
        level : int or str, optional
            The level name or index to filter on (default is 0).
        inplace : bool, optional
            Whether to modify the current object in-place or return a new filtered object (default is True).
        with_copy : bool, optional
            Whether to make a copy of the data whilst filtering, returns a view if False

        Returns
        -------
        None or NestedData
            If `inplace` is True, modifies the current object. Otherwise, returns a new filtered `NestedData`.
        """
        level = self._get_level_idx(level)

        def _recursive_filter(obj, curr_lvl):
            if curr_lvl == level:
                return {k: v for k, v in obj.items() if k in whitelist}
            else:
                for key, val in obj.items():
                    obj[key] = _recursive_filter(val, curr_lvl + 1)
            return obj

        if inplace:
            self._data = _recursive_filter(self._data, 0)
            return None
        else:
            obj = self.new_like(self)
            d = deepcopy(self._data) if with_copy else self._data
            obj._data = _recursive_filter(d, 0)
            return obj

    def transpose_levels(self, reordered_levels, inplace=False):
        """
        Transpose the nested data levels to a new order.

        Parameters
        ----------
        reordered_levels : tuple of str or int
            The new order of the levels, specified by names or indices.
        inplace : bool, optional
            Whether to modify the current object in-place or return a new object (default is False).

        Returns
        -------
        NestedData
            The transposed NestedData (or None if inplace=True).
        """
        reordered_level_indices = [self._get_level_idx(lvl) for lvl in reordered_levels]
        reordered_levels = [self.levels[idx] for idx in reordered_level_indices]

        if set(reordered_levels) != set(self.levels) or len(reordered_levels) != self.nlevels:
            raise ValueError('reordered_levels must be a permutation of current levels')

        def _gather_key_paths(obj, key_path=()):
            if not isinstance(obj, dict):
                yield key_path, obj
            else:
                for key, val in obj.items():
                    yield from _gather_key_paths(val, key_path + (key,))

        reordered_data = {}
        for key_path, val in _gather_key_paths(self._data):
            reordered_path = tuple(key_path[i] for i in reordered_level_indices)
            sd = reordered_data
            for key in reordered_path[:-1]:
                sd = sd.setdefault(key, {})
            sd[reordered_path[-1]] = val

        if inplace:
            self.levels = reordered_levels
            self._data = reordered_data
            return None
        else:
            return NestedData(levels=reordered_levels, dtype=self.dtype, data=reordered_data)

    @classmethod
    def _json_serialise_data(cls, obj, precision):
        if isinstance(obj, dict):
            json_serialisable = {}
            for key, val in obj.items():
                json_serialisable[key] = cls._json_serialise_data(val, precision)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            json_serialisable = []
            for val in obj:
                json_serialisable.append(cls._json_serialise_data(val, precision))
        elif isinstance(obj, (float, np.floating)):
            json_serialisable = float(obj)
            if precision is not None:
                json_serialisable = round(json_serialisable, precision)
        elif isinstance(obj, np.integer):
            json_serialisable = int(obj)
        elif isinstance(obj, (str, int, bool)) or obj is None:
            json_serialisable = obj
        elif isinstance(obj, np.bool_):
            json_serialisable = bool(obj)
        else:
            raise NotImplementedError(f'json serialisation not implemented for type: {type(obj)}')
        return json_serialisable

    def to_json(self, precision=5):
        """
        Serializes the nested data dictionary to a JSON-compatible format.

        Parameters
        ----------
        precision : int, optional
            The number of decimal places to use when serializing floating-point values (default is 5).

        Returns
        -------
        dict
            A JSON-compatible dictionary representing the nested data.
        """
        return {
            'cls': self.__class__.__name__,
            'levels': self.levels,
            'dtype': [dtype.__name__ for dtype in self.dtype],
            'data': self._json_serialise_data(self._data, precision)
        }

    @classmethod
    def _json_deserialise_data(cls, data, nlevels, dtype, encode_method=None):

        def _deserialise(obj, depth):
            if depth == nlevels:
                for dt in dtype:
                    try:
                        return dt(obj)
                    except (ValueError, TypeError):
                        continue
                else:
                    raise ValueError(f'Could not convert object {obj} with dtypes {dtype_converters}')
            else:
                for key, val in obj.items():
                    obj[key] = _deserialise(val, depth + 1)
            return obj

        return _deserialise(data, 0)


    @classmethod
    def from_json(cls, obj, subset=None):
        """
        Creates a `NestedData` instance from a JSON-compatible dictionary.
        The dictionary must have the keys "levels", "dtype", and "data"

        Parameters
        ----------
        obj : dict
            The JSON-compatible dictionary to create the `NestedData` from.

        Returns
        -------
        NestedData
            A new `NestedData` instance created from the JSON data.

        Raises
        ------
        ValueError
            If the JSON object is incorrectly formatted.
        """
        try:
            cls_name = obj.pop('cls')
            encode_method = obj.pop('encode_method', None)
            levels = obj['levels']
            dtype = tuple(DTYPES[d] for d in obj['dtype'])
            data = obj['data']
        except KeyError:
            raise ValueError('nested data json is incorrectly formatted')

        if cls_name != cls.__name__:
            raise ValueError(f'Cannot create {cls.__name__} object from json of class {cls_name}')

        obj['dtype'] = dtype
        if subset is not None:
            data = {key: val for key, val in data.items() if key in subset}
        obj['data'] = cls._json_deserialise_data(data, len(levels), dtype, encode_method)
        return cls(**obj)


class NestedDataArray(NestedData):

    def __init__(self, levels, dtype=(np.ndarray,), data=None):
        if not (dtype == (np.ndarray,) or dtype == np.ndarray):
            raise ValueError("NestedDataArray dtype must be numpy.ndarray")
        super().__init__(levels, dtype=np.ndarray, data=data)

    def __getitem__(self, index):
        if isinstance(index, (str, list, slice, type(Ellipsis))):
            return NestedData.__getitem__(self, index)
        elif isinstance(index, tuple):
            if len(index) <= self.nlevels:
                return NestedData.__getitem__(self, index)

            dct_idx, arr_idx = index[:self.nlevels], index[self.nlevels:]
            selection = NestedData.__getitem__(self, dct_idx)
            if isinstance(selection, np.ndarray):
                return selection[tuple(arr_idx)]

            def _slice_arrays(obj):
                for key, val in obj.items():
                    if isinstance(val, dict):
                        _slice_arrays(val)
                    else:
                        obj[key] = val[tuple(arr_idx)]

            _slice_arrays(selection._data)
            return selection

    def __setitem__(self, index, value):
        if isinstance(index, (str, slice, type(Ellipsis))):
            NestedData.__setitem__(self, index, value)
        elif isinstance(index, tuple):
            if len(index) <= self.nlevels:
                NestedData.__setitem__(self, index, value)
                return

            dct_idx, arr_idx = index[:self.nlevels], index[self.nlevels:]
            try:
                selection = NestedData.__getitem__(self, dct_idx)
            except KeyError:
                key_path = ' -> '.join(dct_idx)
                raise KeyError(f'To set array items, array must already exist at "{key_path}"')
            if isinstance(selection, np.ndarray):
                selection[tuple(arr_idx)] = value
                return

            def _setval_arrays(obj):
                for val in obj.values():
                    if isinstance(val, dict):
                        _setval_arrays(obj)
                    else:
                        val[tuple(arr_idx)] = value

            _setval_arrays(selection._data)

    def stack_values(self):
        stacked = []
        for _, arr in self.deep_items():
            stacked.append(arr)
        return np.asarray(stacked)

    @classmethod
    def _json_serialise_data(cls, obj, precision, encode_method):
        if isinstance(obj, dict):
            json_serialisable = {}
            for key, val in obj.items():
                json_serialisable[key] = cls._json_serialise_data(val, precision, encode_method)
        elif isinstance(obj, np.ndarray):
            if encode_method == 'full':
                json_serialisable = array_encoder_full(obj, precision)
            elif encode_method == 'rle':
                json_serialisable = array_encoder_rle(obj, precision)
            elif encode_method == 'sparse':
                json_serialisable = array_encoder_sparse(obj, precision)
            else:
                raise ValueError(f'encode method "{encode_method}" not recognised')
        else:
            raise ValueError('object is not a numpy.ndarray')
        return json_serialisable

    @classmethod
    def _json_deserialise_data(cls, obj, nlevels, dtype, encode_method='full'):
        if dtype != (np.ndarray,):
            raise ValueError('object is not a json representation of a numpy.ndarray')
        if encode_method == 'full':
            decoder = array_decoder_full
        elif encode_method == 'rle':
            decoder = array_decoder_rle
        elif encode_method == 'sparse':
            decoder = array_decoder_sparse
        else:
            NotImplementedError()

        def _deserialise(obj, depth):
            if depth == nlevels:
                return decoder(obj)
            else:
                for key, val in obj.items():
                    obj[key] = _deserialise(val, depth + 1)
            return obj

        return _deserialise(obj, 0)

    def to_json(self, precision=5, encode_method='full'):
        """
        Serializes the nested data dictionary to a JSON-compatible format.

        Parameters
        ----------
        precision : int, optional
            The number of decimal places to use when serializing floating-point values (default is 5).
        encode_method: None
            Whether to encode the full values or a sparse encoding

        Returns
        -------
        dict
            A JSON-compatible dictionary representing the nested data.
        """
        return {
            'cls': self.__class__.__name__,
            'levels': self.levels,
            'dtype': tuple(dtype.__name__ for dtype in self.dtype),
            'data': self._json_serialise_data(self._data, precision, encode_method),
            'encode_method': encode_method,
        }
