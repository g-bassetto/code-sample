import numpy as np
from pathlib import Path
import threading as _threading
from maprfutils.utils.grid_utils import mk_positive_axis
from maprfutils.utils.grid_utils import mk_symmetric_axis
from maprfutils import distributions as dist

try:
    from ruamel import yaml
except (ImportError, ModuleNotFoundError):
    from pyaml import yaml


class Config:
    _config_lock = _threading.RLock()

    @staticmethod
    def _config_file_name():
        return Path(__file__).parent / 'config.yml'

    def __init__(self):
        self._do_not_load_and_save = dir(self)

    def load(self):
        with self.config_lock:
            if not self._config_file_name().is_file():
                return

            # load settings from file
            with open(self._config_file_name(), 'r') as configdata:
                settings = yaml.load(configdata, Loader=yaml.SafeLoader)

            for group, values in settings.items():
                if group not in self._do_not_load_and_save and not \
                        group.startswith('_'):
                    setattr(self, group.replace('-', '_'), values)

    def save(self):
        with self.config_lock:
            config_data_cpy = self.__dict__.copy()
            for k in list(config_data_cpy.keys()):
                if k in self._do_not_load_and_save or k.startswith('_'):
                    config_data_cpy.pop(k)

            config_data_cpy = {k.replace('_', '-'): v
                               for k, v in
                               config_data_cpy.items()}
            with self._config_file_name().open('w') as stream:
                yaml.dump(config_data_cpy, stream, default_flow_style=False)

    @property
    def config_lock(self):
        return self._config_lock

    @property
    def path(self) -> Path:
        return Path(self.root_path)

    @property
    def path_to_data(self):
        return self.path / 'data'

    @property
    def path_to_figs(self):
        return self.path / 'figs'

    @property
    def path_to_fits(self):
        return self.path / 'fits'

    @property
    def path_to_outs(self):
        return self.path / 'outs'

    @property
    def path_to_tables(self):
        return self.path / 'tables'

    @property
    def path_to_nbs(self):
        return self.path / 'notebooks'

    @property
    def fov_x(self):
        ratio = self.screen_size['nx'] / self.screen_dst
        return np.rad2deg(np.arctan(0.5 * ratio))

    @property
    def fov_y(self):
        ratio = self.screen_size['ny'] / self.screen_dst
        return np.rad2deg(np.arctan(1.0 * ratio))

    @property
    def deg_per_pixel_x(self):
        return 2 * self.fov_x / self.frame_size['nx']

    @property
    def deg_per_pixel_y(self):
        return 1 * self.fov_y / self.frame_size['ny']

    @property
    def AXIS_X(self):
        return mk_symmetric_axis(self.frame_size['nx'], self.deg_per_pixel_x)

    @property
    def AXIS_Y(self):
        return mk_positive_axis(self.frame_size['ny'], self.deg_per_pixel_y,
                                reverse=True)

    @property
    def GRIDS(self):
        return np.meshgrid(self.AXIS_X, self.AXIS_Y)

    @property
    def GRID_X(self):
        return self.GRIDS[0]

    @property
    def GRID_Y(self):
        return self.GRIDS[1]

    @property
    def MAX_RES_X(self):
        return np.abs(np.diff(self.AXIS_X)).mean()

    @property
    def MAX_RES_Y(self):
        return np.abs(np.diff(self.AXIS_Y)).mean()

    @property
    def MAX_FREQ(self):
        return 0.5 / max(self.MAX_RES_X, self.MAX_RES_Y)

    def prior_x(self):
        return dist.Uniform(min(self.AXIS_X), max(self.AXIS_X))

    def prior_y(self):
        return dist.Uniform(min(self.AXIS_Y), max(self.AXIS_Y))

    def prior_f(self):
        return dist.Beta(1.5, 8, (0, self.MAX_FREQ))


del _threading

config = Config()
config.load()
