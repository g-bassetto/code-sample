import click

from maprfanalysis.figures.utils import *
from .schema import *
from ..snr import gen_snr_values

__all__ = ['importer', 'create_model', 'compute_snr']


@click.command('compute-snr')
@click.option('-k', '--keep', 'total', type=click.FLOAT, default=0.95)
@click.argument('fit-id', type=click.STRING, default=None)
def compute_snr(fit_id, total):
    if fit_id is None:
        return

    from maprfanalysis import bbo
    from maprfanalysis.bbo.tasks import BlockLoader

    fit = bbo.fits.loc[fit_id]
    model = bbo.models.get_item_data(fit.model_id)
    frames, spikes = BlockLoader(fit.block_id, fit.index1,
                                 fit.index2).get_data()
    model.set_frames(frames)
    model.set_spikes(spikes)
    del frames, spikes

    samples = load_fit_samples(bbo.fits, fit_id,
                               param_names=get_param_names(model),
                               resample=False)
    samples = samples.sort_values('weights', ascending=False)
    samples = samples.assign(cumw=samples.weights.cumsum()) \
        .query(f'cumw < {total}').drop('cumw', axis=1)
    samples = samples.assign(weights=samples.weights / samples.weights.sum())
    with click.progressbar(gen_snr_values(model, samples), len(samples),
                           'computing snr values') as snr_values:
        snr = pd.DataFrame(list(snr_values)).sort_values('weights')

    snr.to_excel(bbo.fits.get_data_path(fit_id).parents[1] / 'snr.xlsx')


@click.command('create-model')
@click.option('-f', '--family', 'family', default='GABOR',
              type=click.Choice(['GABOR']))
@click.option('-o', '--output', 'output', default='LINEAR',
              type=click.Choice(['LINEAR', 'ENERGY', 'QUADRATIC']))
@click.option('-l', '--invlink', 'invlink', default='EXP',
              type=click.Choice(['EXP', 'SOFTPLUS']))
@click.option('--separable/--non-separable', default=True)
def create_model(family, output, invlink, separable):
    from maprfanalysis.model import ModelBuilder, bbo_priors

    model = ModelBuilder() \
        .set_family(family) \
        .set_output(output) \
        .set_invlink(invlink) \
        .set_separable(separable) \
        .set_priors(bbo_priors) \
        .build()
    try:
        models.insert(family, output, invlink, separable, model=model)
    except KeyError:
        raise
    finally:
        models.to_excel(models.path_to_xlsx)


@click.group('import')
def importer():
    pass


@importer.command('movies')
@click.option('-p', '--path', 'src_path', default='.',
              type=click.Path(exists=True, resolve_path=True))
def import_movies(src_path):
    # This implementation is sub-obtimal (see comment in `schema.Movies`)
    # TODO: move this logic inside Movies.insert()
    # TODO: add a float parameter `rate` to the signature of the command
    # TODO: change `path` to `file`, making the command to import a single
    #  movie by default
    # TODO: keep the loop as an extra, separate option.

    # src_path = Path(src_path)
    # for movie_path in src_path.glob('*.npz'):
    #     movie_name = movie_path.stem
    #
    #     duplicate_name = movie_name in Movies.movie_name.values
    #     duplicate_path = movie_path in Movies.movie_path.values
    #     if duplicate_name or duplicate_path:
    #         click.echo(f'{movie_name!s} already imported: SKIPPING')
    #         continue
    #
    #     click.echo(f'IMPORTING : {movie_name!s}')
    #     movie = dict(**np.load(movie_path))
    #     shape = movie['frames'].shape
    #     rate = float(movie.get('frate', np.nan))
    #
    #     Movies.insert(movie_path, rate, shape)
    #
    # Movies.to_excel(Movies.path_to_xlsx)
    raise NotImplementedError()


@importer.command('blocks')
@click.option('-p', '--path', 'path', default='.',
              type=click.Path(exists=True, resolve_path=True))
def import_blocks(path):
    # TODO: move this logic inside Blocks.insert()

    #     from maprfanalysis import io
    #     from maprfanalysis.utils import GROUP_NAME_TEMPLATE
    #
    #     def display(item):
    #         return 'importing %s' % getattr(item, 'stem', '')
    #
    #     # load saved data in memory
    #     movie_id = Movies.reset_index().set_index('file')['movie_id']
    #
    #     query = sorted(Path(path).glob(GROUP_NAME_TEMPLATE))
    #     with click.progressbar(query, item_show_func=display) as paths:
    #         for filename in paths:
    #             cell = filename.stem
    #
    #             header = io.load_header(filename / 'header.xlsx')
    #             spikes = io.load_spikes(filename / 'spikes.npz')
    #             spikes = pd.DataFrame.from_dict(
    #                 {key: [value] for key, value in spikes.items()},
    #                 orient='index', columns=['data']
    #             )
    #             masks = header[['mask', 'x1', 'y1', 'x2', 'y2']]
    #             header = header\
    #                 .drop(masks.columns, axis=1) \
    #                 .rename(columns={'filename': 'movie_id'})
    #             header = header\
    #                 .drop(set(header.columns) - set(FIELDS), axis=1) \
    #                 .rename(columns={'movie_id': 'movie'})
    #
    #             # merge header and spikes, assign mask theano-filters and rename index
    #             blocks = pd.merge(header, spikes, how='inner', left_index=True,
    #                               right_index=True) \
    #                 .rename(columns={'filename': 'movie'})
    #             blocks = blocks.assign(mask=masks.apply(get_mask_code, axis=1))
    #             blocks.index.rename('block', inplace=True)
    #             # include movie information and remove broken blocks
    #             blocks = blocks.join(movie_id, on='movie').drop(columns=['movie'])
    #
    #             data_nbins = blocks.apply(lambda x: len(x.data), axis=1)
    #             stim_nbins = blocks[['movie_id']]\
    #                 .join(Movies['nt'], on='movie_id')['nt']
    #
    #             blocks = blocks.loc[data_nbins == stim_nbins]
    #
    #             if len(blocks.index) == 0:
    #                 continue
    #             blocks = _add_block_id(blocks.assign(cell=cell).reset_index())
    #
    #             # append new blocks and deal with duplicates
    #             duplicates = Blocks.index.intersection(blocks.index)
    #             NEW_BLOCKS = blocks.loc[blocks.index.difference(duplicates), FIELDS]
    #             DUP_BLOCKS = blocks.loc[duplicates]  # !!! for now discard
    #             for block_id, block in NEW_BLOCKS.iterrows():
    #                 Blocks.loc[block_id] = block
    #
    #
    #     Blocks.to_excel(Blocks.path_to_xlsx)
    #     Movies.to_excel(Movies.path_to_xlsx)

    raise NotImplementedError()
