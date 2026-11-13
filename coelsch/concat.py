from .records.records import BaseRecords
from .utils import load_json


def concatenate_records(record_objs, merge_suffixes=None, suppress_suffixes=False):
    """
    Concatenates multiple record objects, optionally appending unique suffixes
    to cell barcodes to avoid collisions.

    Parameters
    ----------
    record_objs : list
        List of record objects to concatenate.
    merge_suffixes : list or None, optional
        List of suffixes to append to the cell barcodes in each record object.
        If None, a range of integers (1, 2, ..., len(record_objs)) is used.

    Returns
    -------
    object
        A single record object representing the concatenated result.

    Raises
    ------
    ValueError
        If the number of provided suffixes does not match the number of record objects, or
        if the types of the record objects are not valid and the same
    """
    if suppress_suffixes:
        if merge_suffixes is not None:
            raise ValueError(
                '--suppress-suffixes and --merge-suffixes are mutually exclusive'
            )
        merge_suffixes = [None] * len(record_objs)
    elif merge_suffixes is None:
        merge_suffixes = range(1, len(record_objs) + 1)
    elif len(record_objs) != len(merge_suffixes):
        raise ValueError('number of merge suffixes is incorrect')

    concat_records = None
    for robj, sfx in zip(record_objs, merge_suffixes):
        if not isinstance(robj, BaseRecords):
            raise ValueError('One or more objects in record_objs are not valid')
        if not suppress_suffixes:
            robj = robj.add_cb_suffix(sfx, inplace=False)
        if concat_records is None:
            concat_records = robj
        else:
            concat_records.merge(robj, inplace=True)
    return concat_records

def run_concat(json_fn, output_json_fn, merge_suffixes=None, suppress_suffixes=False):
    '''
    Concatenates marker jsons, potentially from different datasets, 
    adding suffixes to cell barcodes to avoid name collisions
    '''
    record_objs = []
    for fn in json_fn:
        robj = load_json(fn, cb_whitelist_fn=None, bin_size=None, data_type='auto')
        record_objs.append(robj)
    concat_records = concatenate_records(record_objs, merge_suffixes, suppress_suffixes)
    concat_records.write_json(output_json_fn)
