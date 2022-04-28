import theano
import theano.tensor as tt
from numpy import array

from ..utils import UpdatesDict
from ..signal.signal import WithChannels
from .pipeline import Operation


sosmat_template = tt.tensor3('sosmat')


def _sosfilt(xi, stages):
    """
    Recursively compute the output of an SOS linear filter
    """
    if len(stages) == 0:
        return xi, UpdatesDict()

    xi, updates = _sosfilt(xi, stages[:-1])
    yi = stages[-1].apply(xi, updates)

    return yi, updates


def _mkpass(i):
    return SOSFilterPass(sosmat_template[i])


class SOSMatrix(WithChannels):
    """
    A matrix representing a series of second order stages.
    """
    def __new__(cls, nsteps, channels=None):
        dtype = (float, 6)
        if channels is not None:
            if isinstance(channels, int):
                channels = [f'ch{i + 1}' for i in range(channels)]
            dtype = [(channel, dtype) for channel in channels]

        return super().__new__(cls, nsteps, dtype)

    def asarray(self):
        return self.view(self.itemtype).reshape(len(self), -1, 6)

    @property
    def a(self):
        """
        Coefficients of the autoregressive part of the filter.
        """
        return self[..., 3:]

    @property
    def b(self):
        """
           Coefficients of the moving-average part of the filter.
        """
        return self[..., :3]

    def operation(self):
        return SOSFilter(self.asarray())


class SOSFilterPass:
    """
    This class represents the application of one second order IIR filter.
    """
    z1 = z2 = None

    def __init__(self, coef):
        self.coef = coef

    @property
    def params(self):
        return [self.coef]

    @property
    def b(self):
        """
           Coefficients of the moving-average part of the filter.
        """
        return self.coef[..., :3].T

    @property
    def a(self):
        """
        Coefficients of the autoregressive part of the filter.
        """
        return self.coef[..., 3:].T

    def apply(self, xi, updates: dict):
        """
        Update the delay units using the Transpose Form II of the filter.
        :param xi: the input signal
        :param updates: a dictionary keeping track of the updates
        :return: the symbolic output of the filter
        """
        yi = self.z1 + self.b[0] * xi

        updates[self.z1] = self.z2 + self.b[1] * xi - self.a[1] * yi
        updates[self.z2] = self.b[2] * xi - self.a[2] * yi

        return yi


class SOSFilter(Operation):

    single_step_mode = False

    @property
    def abstract_params(self):
        return [sosmat_template]

    @property
    def concrete_params(self):
        return [self.sosmat]

    def __len__(self):
        return len(self.stages)

    def __init__(self, sosmat):
        self.sosmat = sosmat
        indices = range(len(sosmat))
        self.stages = [_mkpass(i) for i in indices]

    def __call__(self, signal):
        return self.build_loop(signal)

    def setup(self, signal):
        """
        Setup the filter to operate with an input with a given number of
        dimensions.
        """
        ndim = signal.ndim - (0 if self.single_step_mode else 1)
        template = array(0, dtype=signal.dtype, ndmin=ndim)
        for i, stage in enumerate(self.stages):
            stage.z1 = theano.shared(template.copy(), f'pass{i + 1}.z1',
                                     strict=False)  # , allow_dowcast=True)
            stage.z2 = theano.shared(template.copy(), f'pass{i + 1}.z2',
                                     strict=False)  # , allow_dowcast=True)

    def init(self, signal, template=None):
        """
        Initialization graph.
        """
        updates = UpdatesDict()
        # if operating in single-step mode, we need to use only the first entry
        # of the time series and not the whole signal
        template = signal if self.single_step_mode else signal[0]
        for stage in self.stages:
            # initialize each delay unit with a zeros tensor of the right size
            updates[stage.z1] = tt.zeros_like(template)
            updates[stage.z2] = tt.zeros_like(template)
        return None, updates

    def main(self, signal, template=None):
        """
        Main body graph.
        """
        if self.single_step_mode:
            return self.build_step(signal)
        else:
            return self.build_loop(signal)

    def build_step(self, signal, template=None):
        result, updates = _sosfilt(signal, self.stages)
        return result, updates

    def build_loop(self, signal, template=None):
        result, updates = theano.scan(
            lambda xi: _sosfilt(xi, self.stages),
            sequences=signal,
            outputs_info=[None]
        )

        return result, updates


if __name__ == '__main__':
    # Invoking this module as a script will run a series of tests.
    # On the long run we should refactor this theano-filters using proper unit testing.

    sm = SOSMatrix(4, ('A', 'B'))

    sm['A'].a[:] = [1, 2, 3]
    sm['A'].b[:] = [4, 5, 6]
    sm['B'].a[:] = [7, 8, 9]
    sm['B'].b[:] = [10, 11, 12]

    print(sm.view(sm.itemtype))
    print(sm.asmatrix())

    sm = SOSMatrix(4)
    sm.a[:] = [1, 2, 3]
    sm.b[:] = [4, 5, 6]

    print(sm.view(sm.itemtype))
    print(sm.asmatrix())
