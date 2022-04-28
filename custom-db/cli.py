import click
from pathlib import Path
from maprfanalysis._cli import cli
from . import config
from .commands import compute_snr


def create_folders(cfg):
    for _, path in cfg.items():
        Path(path).mkdir(parents=True, exist_ok=True)


@cli.group()
def bbo():
    pass


bbo.add_command(compute_snr)


@bbo.command('setup')
def setup():
    from maprfanalysis import config

    install_path = Path.cwd() / 'bbo'

    bbo_config = {
        'data-path': str(install_path / 'data'),
        'outs-path': str(install_path / 'outs'),
        'tables-path': str(install_path / 'tables'),
    }
    config.bbo['paths'].update(bbo_config)
    create_folders(bbo_config)

    config.save()


@bbo.command('run-fits')
def run_fits():
    import luigi
    import maprfanalysis.bbo as bboutils

    # excluded = bboutils.models.query('output=="ENERGY"').index
    fits = bboutils.fits.query('model_id not in @excluded')

    tasks = [bboutils.AnalysisTask(fit_id) for fit_id in fits.index]
    luigi.build(tasks)
