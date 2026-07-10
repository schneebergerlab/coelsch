import logging
from collections import Counter
import numpy as np

from coelsch.load.barcodes import CellBarcodeWhitelist
from coelsch.records import MarkerRecords, PredictionRecords


log = logging.getLogger('coelsch')


def load_json(json_fn, cb_whitelist_fn, bin_size, data_type='markers', subset=None, frozen=False):
    """
    Load MarkerRecords or PredictionRecords from a JSON file and filter based on cell barcode whitelist.

    Parameters
    ----------
    json_fn : str
        Path to the JSON file containing the data.
    cb_whitelist_fn : str, optional
        Path to the file containing the cell barcode whitelist.
    bin_size : int, optional
        The bin size used for processing the data.
    data_type : {'markers', 'predictions', 'auto'}, optional
        Specifies the type of data to load. The default is 'markers'.
    subset : iterable, optional
        A subset of the data to load.

    Returns
    -------
    MarkerRecords or PredictionRecords
        The data loaded from the JSON file, filtered based on the barcode whitelist.
    
    Raises
    ------
    ValueError
        If the bin size does not match the bin size in the JSON file or if no barcodes match
        the whitelist.
    NotImplementedError
        If the data type is not recognized.
    IOError
        If the data type cannot be automatically determined from the file.
    """
    if data_type == 'markers':
        data = MarkerRecords.read_json(json_fn, subset=subset, frozen=frozen)
    elif data_type == 'predictions':
        data = PredictionRecords.read_json(json_fn, subset=subset, frozen=frozen)
    elif data_type == 'auto':
        try:
            data = MarkerRecords.read_json(json_fn, subset=subset, frozen=frozen)
        except ValueError as exc:
            try:
                data = PredictionRecords.read_json(json_fn, subset=subset, frozen=frozen)
            except ValueError:
                raise IOError(
                    f'data type of file {json_fn} could not be determined automatically'
                ) from exc
    else:
        raise NotImplementedError()
    if bin_size is not None and data.bin_size != bin_size:
        raise ValueError('"--bin-size" does not match bin size specified in json-fn, '
                         'please modify cli option or rerun previous coelsch steps')
    log.info(f'Read {len(data)} cell barcodes from json file {json_fn}')
    if cb_whitelist_fn:
        cb_whitelist = read_cb_whitelist(cb_whitelist_fn,  validate_barcodes=False).toset()
        data.filter(cb_whitelist)
        if not data:
            raise ValueError('No CBs from --cb-whitelist-fn are present in json-fn')
        log.info(f'{len(data)} barcodes remain after cb whitelist filtering')
    return data


def validate_ploidy(data, ploidy_type):
    if ploidy_type is None:
        if data.ploidy_type is not None:
            ploidy_type = data.ploidy_type
        else:
            # use haploid as fallback default
            ploidy_type = 'haploid'
    elif data.ploidy_type != ploidy_type:
        log.warn(f'Recorded ploidy type of data is "{data.ploidy_type}", but specified '
                 f'ploidy is "{ploidy_type}". This may not be intended.')
    data.ploidy_type = ploidy_type
    return ploidy_type


def spawn_child_rngs(rng):
    """
    Spawn a series of random number generators (RNGs) based on a given parent RNG.

    This function generates child random number generators using the state of a parent RNG.
    Each child RNG is independent but has a state determined by the parent. Useful for
    retaining deterministic behaviour when using multiprocessing.

    Parameters
    ----------
    rng : np.random.Generator
        The parent random number generator to base child RNGs on.

    Yields
    ------
    np.random.Generator
        A child random number generator, created from the parent RNG.
    """
    while True:
        yield np.random.default_rng(rng.spawn(1)[0])
