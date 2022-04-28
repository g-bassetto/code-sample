import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Union, Tuple
from random import getrandbits


from .config import config as settings
from maprfanalysis.database.hashing import HexHash as _HexHash
from maprfanalysis.database.builder import Database, as_method, Field
import re

__all__ = ['models', 'movies', 'blocks', 'runs', 'fits', 'db',
           'is_valid_block_id', 'cell_from_block_id', 'DuplicateEntryException']

CELL_ID_TEMPLATE = re.compile('R\d{6}ct\d{2}-cell\d{2}')
BLOCK_ID_TEMPLATE = re.compile('R\d{6}ct\d{2}-cell\d{2}-[a-z]')

MASK_TUPLE_TYPE = Union[int, Tuple[int, int, int, int]]


def cell_from_block_id(s):
    return CELL_ID_TEMPLATE.match(s).group()


def is_valid_block_id(s):
    return bool(BLOCK_ID_TEMPLATE.fullmatch(s))


class DuplicateEntryException(ValueError):
    pass


class DatabaseBBO(Database):
    def __init__(self):
        super().__init__({})
    
    @property
    def path_to_tables(self):
        return settings.path_to_tables

    @property
    def path_to_data(self):
        return settings.path_to_data

    @property
    def path_to_outs(self):
        return settings.path_to_outs


db = DatabaseBBO()


@db.table(data=True, folder='data/')
class Blocks:
    block_id: str = Field(is_key=True)
    cell: str = Field(required=True)
    block: str = Field(required=True)
    shutter: str
    angle: float = Field(required=True)
    mask: int  # TODO: rename -> mask_id
    movie_id: str = Field(required=True)

    compute_record_id = as_method(lambda r: '%s-%s' % (r.cell, r.block))

    def get_spikes(self, block_id: str) -> np.ndarray:
        return np.load(str(self.get_data_path(block_id)))

    def get_frames(self, block_id: str) -> np.ndarray:
        movie_id = self.loc[block_id].movie_id
        return movies.get_item_data(movie_id)

    def insert(self, block_id: str, movie_id: str, angle: float,
               shutter: str = None, mask: MASK_TUPLE_TYPE = None) -> pd.Series:
        """
        Insert a new recording block in the database.
        Args:
            block_id (str): the ID used to identify the recording.
            movie_id (str): the ID of the stimulus movie used for this
                recording, as it appears in `Movies.index`.
            angle (float): the tilting angle, in degrees, of the experimental
                platform for this recording.
            shutter (str): the strings 'L', 'R' or `None`; it indicates which
                eye-shutter was used for this recording; defaults to `None`.
            mask (int, tuple): tuple of 4 `int`s, `int`, or `None`; the 4
                integers represent the (left, right, top, bottom) coordinates
                of the rectangular mask applied to the stimulus during this
                recording; if `int`, it contains the compressed
                representation of the mask used; use `None` if no mask was
                used; defaults to `None`.
        Returns (pd.Series):
            The newly inserted record.
        """
        raise NotImplementedError()

    def get_data_path(self, block_id: str) -> Path:
        return settings.path_to_data / 'blocks' / (block_id + '.npy')

    def get_item_data(self, block_id: str) -> tuple:
        x = self.get_frames(block_id)
        y = self.get_spikes(block_id)
        return x, y


@db.table(data=True, data_fmt='npz', folder='data/')
class Movies:
    movie_id: str = Field(is_key=True)
    rate: float = Field(required=True)
    nx: int
    ny: int
    nt: int
    movie_path: str
    movie_name: str = Field(required=True)

    compute_record_id = _HexHash('movie_name')

    def insert(self, filename: str, rate: float, shape: tuple = None) -> pd.Series:
        """
        Insert a new movie in the database.
        Args:
            filename (str): name of the file containing the movie's data.
            rate (float): frame rate of the movie, in Hertz.
            shape (tuple): (nt, ny, nx), the shape of the movie data; if `None`
                it is computed from the movie data defaults to `None`.

        Returns (pd.Series):
            The newly inserted record.
        """
        # TODO: this implementation is sub-otpimal, as it implies that the
        #  data has already been moved to its final destination. Deal here
        #  with the importing logic, so that the command in the `commands`
        #  module becomes a simple wrapper.
        raise NotImplementedError()

    def get_data_path(self, movie_id: str) -> Path:
        filename = self.loc[movie_id].movie_name + '.npy'
        return settings.path_to_data / 'movies' / filename

    def get_item_data(self, movie_id: str, fmt: str=None) -> np.ndarray:
        data = np.load(self.get_data_path(movie_id))
        if fmt in ('float', float):
            from ..utils import convert_uint_frames_to_float
            data = convert_uint_frames_to_float(data)
        return data


@db.table(data=True, folder='data/')
class Models:
    model_id: str = Field(is_key=True)
    family: str = Field(required=True)
    output: str = Field(required=True)
    invlink: str = Field(required=True)
    separable: bool = Field(required=True)

    compute_record_id = _HexHash('famlily', 'output', 'invlink', 'separable')

    def insert(self, family: str, output: str, invlink: str, separable: bool,
               *, model_id: str = None, model=None) -> pd.Series:
        # TODO: write docstring
        record = (family, output, invlink, separable)
        if model_id is None:
            model_id = self.compute_record_id(record)
        if model_id in self.index:
            raise KeyError("A model with this name already exist, "
                           "and inserting a new one will override the former "
                           "model's data file, thus invalidating any result "
                           "in `Runs` or `Fits` referring to this model.")
        if model is not None:
            with self.get_data_path(model_id).open('wb') as file:
                pickle.dump(model, file)

        self.loc[model_id] = record
        return self.loc[model_id]

    def get_data_path(self, model_id: str) -> Path:
        return settings.path_to_data / 'models' / (model_id + '.pkl')

    def get_item_data(self, model_id: str) -> object:
        with open(self.get_data_path(model_id), 'rb') as file:
            return pickle.load(file)


@db.table
class Runs:
    fit_id: str = Field(is_key=True)
    nlive: int = Field(required=True)
    seed: int = Field(required=True)
    label: str = Field(required=False)

    compute_record_id = _HexHash('fit_id', 'nlive', 'seed')

    @staticmethod
    def get_results_path(item):
        return fits.get_data_path(item.name[0]) / (item.label + '.npz')

    @staticmethod
    def is_run_completed(item):
        return item.results_path.exists()

    def compute_run_index(self):
        return self.assign(val=1).groupby('fit_id').cumsum()['val']

    def __on_load__(self):
        index = self.compute_run_index()
        self.table = self\
            .assign(run_id=index)\
            .reset_index()\
            .set_index(['fit_id', 'run_id'])

    def get_data_path(self, run_id: str) -> Path:
        fit_id = self.loc[run_id].fit_id
        return fits.get_data_path(fit_id) / (run_id + '.npz')

    def insert(self, fit_id: str, nlive: int, seed: int, label: str = None)\
            -> pd.Series:
        """
        Insert a new record in the table.

        Args:
            fit_id (str): the ID of the fit, as it appears in `Fits`, the new
                run is associated to.
            nlive (int): the number of live points used by the nested sampler.
            seed (int): the seed used to initialize the random number
                generator used by the nested-sampling routine.
            label (str): the ID to associate to the new record. If `None` it
                will be computed based on the value of the other inputs.
                Defaults to `None`.

        Returns:
            The newly inserted record.

        Raises:
            ValueError: if the combination (`nlive`, `seed`) already exist
            for `fit_id`.
        """

        records = self.query('fit_id == @fit_id')
        while seed is None:
            # keep generating seeds until we found one that was not used
            # in an other record with the same combination of the other
            # parameters
            candidate = getrandbits(32)
            if records.query('nlive == @nlive & seed == @seed').empty:
                seed = candidate

        if not records.query('nlive == @nlive & seed == @seed').empty:
            raise ValueError(f"Cannot insert a new run ({nlive}, {seed}) for "
                             f"fit {fit_id} because such an entry already "
                             f"exist.")
        if label is None:
            label = self.generate_new_label(fit_id, nlive, seed)

        run_id = len(records) + 1  # not ideal if runs can also be deleted
        self.loc[(fit_id, run_id), :] = (nlive, seed, label)
        return self.loc[(fit_id, run_id)]

    def generate_new_label(self, fit_id, nlive, seed):
        records = self.query('fit_id == @fit_id')
        label = hash((fit_id, nlive, seed)) >> 32
        while label in records.label:
            # keep adding noise until a new, unused label for this group is
            # generated
            label ^= getrandbits(32)
        return hex(label)[2:]

    def get_block_data(self, run_id: str) -> tuple:
        """
        Return the model input data for a specific run.

        Args:
            run_id (str): the ID of the run as it appears in `Runs`.

        Returns tuple:
            A tuple (`x`, `y`), where `x` is the stimulus applied to the
            neuron and `y` is the binned sequence of spike counts.
        """
        f_id = self.loc[run_id, 'fit_id']
        return fits.get_block_data(f_id)

    def get_results(self, run_id):
        from maprfutils import load_results
        return load_results(self.get_results_path(run_id))


@db.table(folder='outs/')
class Fits:
    fit_id: str = Field(is_key=True)
    model_id: str = Field(required=True)
    block_id: str = Field(required=True)
    index1: int = Field(required=True)
    index2: int = Field(required=True)
    sampler: str = Field(required=True)
    log_z: float = Field(required=False)

    def __on_load__(self):
        self.table = self.table\
            .assign(
                nruns=runs.groupby('fit_id').size(),
                nlive=runs[['nlive']].groupby('fit_id').sum()
            )

    def insert(self, model_id: str, block_id: str, slice_id: tuple,
               sampler: str, fit_id: str = None) -> pd.Series:
        """
        Insert a new record in the table.

        Args:
            model_id (str): the ID of the model to use for this fit,
                as it appears in `Models`.
            block_id (str): the ID of the block used for this fit,
                as it appears in `Blocks`.
            slice_id (tuple): (start, end).
            sampler (str): the type of sampler to use to sample the posterior
                over the model parameters.
            fit_id (str): the ID to associate to the new record. If `None` it
                will be computed based on the value of the other inputs.
                Defaults to `None`.

        Returns (pd.Series):
            The newly inserted record.
        """

        # TODO: handle this input argument properly
        i1, i2 = slice_id  # ok for the moment

        record = (model_id, block_id, i1, i2, sampler, np.nan)
        query_str = 'model_id==@model_id & block_id==@block_id & ' +\
                    'index1==@i1 & index2==@i2 & sampler==@sampler'
        if not self.query(query_str).empty:
            raise DuplicateEntryException(str(record))

        if fit_id is None:
            # TODO: handle the rare case of conflicting hashes
            # compute fit_id based on the other fields
            fit_id = hex(np.uint64(hash(record)))[2:]

        logging.info(f'inserting {record} -> {fit_id}')
        self.loc[fit_id] = record
        if runs.query('fit_id == @fit_id').empty:
            self.add_run(fit_id)

        return self.loc[fit_id]

    def add_run(self, fit_id: str, nlive: int = None, seed: int = None) -> \
            pd.Series:
        """
        Create a new run nested-sampling run to a specific fit.

        Args:
            fit_id (int): the fit ID as it appears in `Fits`.
            nlive (int): the number of live points to use for the
                nested-sampling run; if `None`, it will be computed based on the
                number of samples of the model and of the sampler used for this
                fit. Defaults to `None`.
            seed (int): the seed used to initialize the random number
                generator used by the nested-sampling routine; if `None`, a
                random seed will be generated. Defaults to `None`.

        Returns (pd.Series):
            The newly created run.
        """
        from maprfutils.sampling.samplers import get_min_n_live_points

        _runs = runs.query('fit_id == @fit_id')
        m_id, b_id, i1, i2, samp = self.loc[fit_id]
        model = models.get_item_data(m_id)
        if nlive is None:
            nlive = get_min_n_live_points(model, samp)

        return runs.insert(fit_id, nlive, seed)

    def get_data_path(self, fit_id: str) -> Path:
        return settings.path_to_outs / 'fits' / self.loc[fit_id].block_id / \
               block_length_str[fit_id] / f'{fit_id}.fit'

    def get_block_data(self, fit_id: str) -> tuple:
        """
        Return the model input data for a specific fit.

        Args:
            fit_id (str): the ID of the fit as it appears in `Fits`.

        Returns tuple:
            A tuple (`x`, `y`), where `x` is the stimulus applied to the
            neuron and `y` is the binned sequence of spike counts.
        """
        record = self.loc[fit_id]
        x, y = blocks.get_item_data(record.block_id)
        i1, i2 = record[['index1', 'index2']]
        return x[i1:i2], y[i1:i2]

    def get_results(self, fit_id: str) -> dict:
        """
        Return the samples for a particular fit. If the fit consists of more
        than one nested-sampling run, the runs are merged into an equivalent,
        global run.

        Args:
            fit_id (str): the ID of the fit as it appers in `Fits`.

        Returns (dict):
            A dictionary containing the output generated by the
            nested-sampler; if the fit consists of more than one run,
            the runs are combined into an equivalent, global run.
        """
        from maprfutils import load_results
        _runs = runs.query(f'fit_id == {fit_id!r}')
        partials = list()
        for run_id in _runs.index:
            try:
                results = load_results(runs.loc[run_id].results_path)
            except FileNotFoundError:
                continue
            else:
                partials.append(results)

        if len(partials) == 0:
            return None
        if len(partials) == 1:
            return partials[0]
        else:
            from dynesty.utils import merge_runs
            return merge_runs(partials)


def get_block_length_str(item):
    return f"{item['min']}m" + (f"{item['sec']:02d}" if item.sec else "")


def make_fit_block_lenght_info():
    rate_info = blocks.join(movies.table, on='movie_id')['rate']
    info = fits.join(rate_info, on='block_id')
    block_length_min = (info.index2 - info.index1) // (info.rate * 60)
    block_length_sec = (info.index2 - info.index1) % (info.rate * 60)
    block_length = pd.DataFrame(
        {'min': block_length_min, 'sec': block_length_sec})

    return block_length.apply(get_block_length_str, axis=1)


models = Models(load_or_create=True)
blocks = Blocks(load_or_create=True)
movies = Movies(load_or_create=True)
runs = Runs(load_or_create=True)
fits = Fits(load_or_create=True)
block_length_str = make_fit_block_lenght_info()


runs_paths = runs.apply(runs.get_results_path, axis=1)
runs.table = runs.assign(results_path=runs_paths)
