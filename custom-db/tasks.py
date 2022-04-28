import numpy as np
import pandas as pd
import luigi
from luigi import Task, LocalTarget
from ruamel import yaml

from pandas import Series

from maprfanalysis.utils import convert_uint_frames_to_float
from maprfanalysis.bbo.queries import *
from tqdm.autonotebook import tqdm

from maprfutils.tasks.common import ModelFactory, DataFactory
from maprfutils.tasks.common import TrainModel, SamplerFactory


class ModelLoader(ModelFactory, luigi.ExternalTask):
    model_id: str = luigi.Parameter()

    def complete(self):
        return self.model_id in models.index

    def get_model(self):
        mdl = get_model(self.model_id)
        mdl.output.delta = mdl.filter.tf.step_size
        return mdl


class BlockLoader(DataFactory, luigi.ExternalTask):
    block_id: str = luigi.Parameter()
    index1: int = luigi.IntParameter()
    index2: int = luigi.IntParameter()

    @property
    def slice(self):
        return slice(self.index1, self.index2)

    def complete(self):
        return self.block_id in blocks.index

    def get_data(self):
        x = self.get_frames()
        y = self.get_spikes()
        return x, y

    def get_spikes(self):
        return get_spikes(self.block_id)[self.slice]

    def get_frames(self):
        x = blocks.get_frames(self.block_id)[self.slice]
        return convert_uint_frames_to_float(x, 'f4')


class PerformFit(luigi.Task):
    fit_id: str = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        fit = fits.loc[self.fit_id]
        self._get_model_task = ModelLoader(fit.model_id)
        self._get_block_task = BlockLoader(fit.block_id, fit.index1, fit.index2)
        self._sampler_factory = SamplerFactory(fit.sampler)

        path = fits.get_data_path(self.fit_id)
        self.header_target = LocalTarget(path / f'{self.fit_id}.header.yml')
        self.samples_target = LocalTarget(path / 'samples.npz',
                                          format=luigi.format.Nop)
        self.merge_info_target = LocalTarget(path / 'merge-info.txt')

    @property
    def results_path(self):
        parts = runs.loc[self.fit_id].squeeze()
        if isinstance(parts, Series):
            return parts.results_path
        else:
            return self.samples_target.path

    def requires(self):
        for _, run in runs.loc[self.fit_id].iterrows():
            yield TrainModel(
                str(run.results_path),
                m_factory=self._get_model_task,
                d_factory=self._get_block_task,
                s_factory=self._sampler_factory,
                nlive=run.nlive,
                seed=run.seed
            )

    @property
    def parts_completed(self):
        parts = runs.loc[self.fit_id]
        return all(parts.apply(runs.is_run_completed, axis=1))

    @property
    def merge_completed(self):
        try:
            with self.header_target.open('r') as stream:
                header = yaml.load(stream, Loader=yaml.Loader)
        except FileNotFoundError:
            return False

        parts = runs.loc[self.fit_id]
        return all((label in header['runs']) for label in parts.label)

    @property
    def requires_merge(self):
        return self.is_multi_run and not self.merge_completed

    @property
    def is_multi_run(self):
        return len(runs.loc[self.fit_id]) > 1

    def complete(self):
        return self.parts_completed and self.merge_completed and \
               self.header_target.exists()

    def run(self):
        parts = runs.loc[self.fit_id]
        if self.requires_merge:
            self.merge_runs(parts)

        header = {k: (v if isinstance(v, str) else v.item())
                  for k, v in fits.loc[self.fit_id].items()}

        with np.load(self.results_path, mmap_mode='r') as file:
            header['logz'] = float(file['logz'][-1])
        header['runs'] = parts.label.to_list()

        with self.header_target.open('w') as file:
            yaml.dump(header, file, default_flow_style=False)

    def merge_runs(self, runs):
        from maprfutils import load_results
        from maprfutils import save_results
        from dynesty.utils import merge_runs

        partial = map(load_results, runs.results_path)
        results = merge_runs(list(partial), False)
        # save samples.npz
        save_results(self.samples_target.path, results)
        # save runs_list.txt
        runs_names = runs.label.to_list()
        with self.merge_info_target.open('w') as file:
            file.writelines('\n'.join(runs_names))

        return results


class CollectFitsInformation(luigi.Task):

    def complete(self):
        log_z = getattr(fits, 'log_z', None)
        if log_z is not None:
            return not log_z.isna().any()
        else:
            return False

    def run(self):
        if not ('log_z' in fits.columns):
            fits.table = fits.assign(log_z=np.nan)
        need_processing = fits.query('log_z.isnull()')
        entries = tqdm(need_processing.index, total=len(need_processing))

        for fit_id in entries:
            fit_task = PerformFit(fit_id)
            yield fit_task
            with fit_task.header_target.open('r') as stream:
                info = yaml.load(stream, Loader=yaml.Loader)
                fits.loc[fit_id, 'log_z'] = info['logz']

        fits.reset_index().drop(columns=['nruns', 'nlive'])\
            .to_excel(fits.path_to_xlsx)


class TaskGroup(luigi.WrapperTask):
    task_id: str = luigi.Parameter()
    task_type: Task = luigi.TaskParameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_list = list()

    def requires(self):
        return self.task_list


class EntropyFunction(luigi.ExternalTask):

    def __init__(self, *args, **kwargs):
        from .. import poissent
        poissent.set_approx_order(3)
        self._entropy = poissent.lower_bound
        super().__init__(*args, **kwargs)

    @property
    def entropy(self):
        return self._entropy

    def complete(self):
        return True


class ComputeEffectiveFreAndSNR(luigi.Task):
    ent_fun_getter = EntropyFunction()
    fit_id: str = luigi.Parameter()
    sampling_task: PerformFit = luigi.TaskParameter()

    def output(self):
        path = fits.get_data_path(self.fit_id) / 'fre-and-snr.xlsx'
        return luigi.LocalTarget(path, format=luigi.format.Nop)

    def requires(self):
        return self.sampling_task

    def dkl(self, r, r0):
        return (r0 - r) - r * (np.log(r0) - np.log(r))

    def run(self):
        from tqdm import tqdm
        results = fits.get_results(self.fit_id)
        model = self.sampling_task._get_model_task.get_model()
        data_loader = self.sampling_task._get_block_task
        model.set_frames(data_loader.get_frames())
        model.set_spikes(data_loader.get_spikes())
        dt = model.output.delta
        invlink = model.output.f

        output = list()
        ent = self.ent_fun_getter.entropy
        for i, samp in enumerate(tqdm(results.samples)):
            model.set_params(samp)
            r0 = dt * invlink(model.output.beta[-1])
            r = dt * invlink(model.output.activation)

            num = self.dkl(r, r0).mean()
            den = ent(r).mean()
            estimates = {
                'fre': r.mean(),
                'snr': 20 * np.log10(num / den),
            }
            output.append(estimates)
        output = pd.DataFrame(output).assign(weights=results.weights)
        output.to_excel(self.output().path, index=False)


class AnalysisTask(luigi.WrapperTask):
    fit_id: str = luigi.Parameter()
    postprocess: list = list()

    @property
    def priority(self):
        return int(100 - 10 * fits.loc[self.fit_id].nruns)

    def requires(self):
        required = list()
        main_task = PerformFit(self.fit_id)
        required.append(main_task)
        for task in self.postprocess:
            required.append(task(self.fit_id, main_task))
        return required
