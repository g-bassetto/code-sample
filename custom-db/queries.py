from numpy.random.mtrand import RandomState
from functools import wraps
from .schema import *
from .schema import is_valid_block_id


@wraps(models.get_item_data)
def get_model(model_id):
    return models.get_item_data(model_id)


@wraps(blocks.get_spikes)
def get_spikes(block_id):
    return blocks.get_spikes(block_id)


@wraps(blocks.get_frames)
def get_frames(block_id):
    return blocks.get_frames(block_id)


def get_block_data(b_id: str) -> tuple:
    """
    Load spikes and frames for a given `block` or `segment` of data.

    Args:
        b_id (str): a block's or a segment's identifier.

    Returns tuple:
        A tuple (x, y), where x and y are both instances of `numpy.ndarray' of
        type `numpy.uint8` and contains, respectively, the frames and the
        spikes of the requested block of data.
    """
    if is_valid_block_id(b_id):
        return blocks.get_item_data(b_id)
    else:
        raise KeyError('`b_id` is not a valid block identifier.')


def get_equally_weighted_samples(r_id, seed=None):
    """
    Load and re-weight a nested-sampling run, so that all samples have the
    same wright.

    Args:
        r_id (str): the run identifier.
        seed (int): a random seed, for reproducibility.
    Returns np.ndarray:
        A 2D ndarray, where columns correspond to variables and rows to samples.
    """
    from numpy import exp
    from dynesty.utils import resample_equal

    if r_id in fits.index:
        results = fits.get_results(r_id)
    else:
        results = runs.get_results(r_id)

    r_state = RandomState(seed) if seed is not None else None
    weights = exp(results.logwt - results.logz[-1])
    samples = resample_equal(results.samples, weights, r_state)

    return samples


def get_fit_info(fit_id):
    """Returns the list of runs corresponding to a given."""
    return runs.groupby('fit_id').get_group(fit_id)


@wraps(fits.get_results)
def get_fit_results(*args):
    return fits.get_results(*args)


@wraps(runs.get_results)
def get_run_results(*args):
    return runs.get_results(*args)
