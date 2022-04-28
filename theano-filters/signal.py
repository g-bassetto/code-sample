from abc import ABCMeta, abstractmethod
import numpy as np


class AbstractSignalOp(metaclass=ABCMeta):
    """
    Defines an abstract operation that can be performed on a signal.

    This class is designed having in mind a specific set of operation,
    namely (nonlinear) filters.
    """

    def mc_signal_mc_kernel(self, signal, kernel, output=None) -> np.ndarray:
        """
        Apply the operation on a multi-channel signal using a multi-channel 
        kernel.
        :var signal: the signal we want to transform
        :var kernel: the kernel we want to apply
        :var output: a buffer where to store the output result

        Signal and kernel must have the same channels. Channels are paired.
        """
        if output is None:
            output = np.zeros_like(signal)
        for channel in signal.channels:
            self.sc_signal_sc_kernel(signal[channel], kernel[channel],
                                     output[channel])
        return output

    def mc_signal_sc_kernel(self, signal, kernel, output=None) -> np.ndarray:
        """
        Apply the operation on a multi-channel signal using a single-channel
        kernel.
        :var signal: the signal we want to transform
        :var kernel: the kernel we want to apply
        :var output: a buffer where to store the output result

        The single kernel channel is broadcasted to the each channel of the
        signal.
        """
        if output is None:
            output = np.zeros_like(signal)
        for channel in signal.channels:
            output[channel] = self.sc_signal_sc_kernel(signal[channel], kernel)
        return output

    def sc_signal_mc_kernel(self, signal, kernel, output=None) -> np.ndarray:
        """
        Apply the operation on a single-channel signal using a multi-channel
        kernel.
        :var signal: the signal we want to transform
        :var kernel: the kernel we want to apply
        :var output: a buffer where to store the output result

        Each channel of the kernel is applyed separately to the input.
        """
        if output is None:
            output = Signal(None, signal.shape, channels=kernel.channels)
        for channel in kernel.channels:
            output[channel] = self.sc_signal_sc_kernel(signal, kernel[channel])
        return output

    def sc_signal_sc_kernel(self, signal, kernel, output=None) -> np.ndarray:
        """
        Apply the operation on a single-channel signal using a single-channel
        kernel.
        :var signal: the signal we want to transform
        :var kernel: the kernel we want to apply
        :var output: a buffer where to store the output result
        """
        if output is None:
            output = np.zeros_like(signal)
        output[:] = self.__impl__(signal, kernel)
        return output

    @abstractmethod
    def __impl__(self, signal, kernel):
        """
        Concrete implementation of the operation represented by this class.

        :var signal: the signal we want to transform
        :var kernel: the kernel we want to apply
        :return: the result of the filtering operation
        """
        raise NotImplementedError()

    def __call__(self, signal, kernel, output=None):
        """
        Apply the operation parameterized by kernel on a signal.
        """
        if bool(signal.channels) and bool(kernel.channels):
            if signal.channels != kernel.channels:
                raise ValueError
            output = self.mc_signal_mc_kernel(signal, kernel, output)
        elif signal.channels:
            output = self.mc_signal_sc_kernel(signal, kernel, output)
        elif kernel.channels:
            output = self.sc_signal_mc_kernel(signal, kernel, output)
        else:
            output = self.sc_signal_sc_kernel(signal, kernel, output)

        return output


def channel_parser(channels):
    """
    Parse channels information. If needed, automatically generate a set of
    channel names.
    :param channels: a list of channel names or the derired number of channels
    :return: a list of channel names
    """
    if channels is not None:
        if isinstance(channels, int):
            channels = [f'ch{i + 1}' for i in range(channels)]
    return channels


class WithChannels(np.ndarray):
    """
    A sublass of numpy.ndarray representing a multi-channel object.

    It is equivalent to a struct array where all fields are constrained to
    have the same dtype.
    """

    @property
    def channels(self) -> tuple:
        """
        The names of the channels of this object
        :return: a tuple containing the channel names
        """
        if self.dtype.isbuiltin:
            return ()
        else:
            return tuple(self.dtype.names)

    @property
    def nchannels(self) -> int:
        """
        The number of channels.
        """
        return len(self.channels) if self.channels else 1

    @property
    def itemtype(self):
        """
        The underlying type of the object.
        """
        if self.dtype.isbuiltin:
            return self.dtype.base
        else:
            return self.dtype[0].base


class Signal(WithChannels):
    """

    """
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)

    def __new__(cls, name, shape, channels=None, dtype=float, buffer=None):
        """
        Construct a new signal with the desired properties.
        :var name:
        :var shape:
        :var channels:
        :var dtype:
        :var buffer:
        """
        if buffer is not None:
            itemtype = buffer.dtype
        else:
            itemtype = dtype

        if channels is not None:
            if isinstance(channels, int):
                channels = [f'ch{i + 1}' for i in range(channels)]
            dtype = [(channel, itemtype) for channel in channels]
        else:
            dtype = itemtype

        array = super().__new__(cls, shape, dtype, buffer)
        array.name = name

        return array


class TimeSeries(Signal):

    @property
    def n_frames(self) -> int:
        """
        The number of points in the time series.
        """
        return self.shape[0]

    @property
    def frame_size(self) -> tuple:
        """
        The shape of each point.
        """
        return self.shape[1:]


def as_signal(name: str, array: np.ndarray, channels: tuple=None,
              cls: type=Signal) -> Signal:
    """
    Wraps an array around a Signal object.
    :param name: the name of the output signal
    :param array: the native array to wrap
    :param channels: the channels of the signal, if any
    :param cls: the specific Signal subclass to use
    :return: a Signal instance wrapping the array
    """
    if channels is not None:
        if isinstance(channels, int):
            channels = [f'ch{i + 1}' for i in range(channels)]
        assert array.shape[-1] == len(channels)

        dtype = [(channel, array.dtype) for channel in channels]
        # the last dimension is absorbed by the channels, so we must remove it
        shape = array.shape[:-1]
    else:
        dtype = array.dtype
        shape = array.shape

    signal = array.view(dtype).reshape(shape).view(cls)
    signal.name = name
    return signal


if __name__ == '__main__':
    # Invoking this module as a script will run a series of tests.
    # On the long run we should refactor this theano-filters using proper unit testing.

    # tests
    x1 = Signal('x1', 10)
    assert x1.shape == (10,)
    assert x1.dtype == float
    assert x1.itemtype == float
    assert not bool(x1.channels)

    x2 = Signal('x2', 10, dtype=int)
    assert x2.shape == (10,)
    assert x2.dtype == int
    assert x2.itemtype == int
    assert not bool(x2.channels)

    x3 = Signal('x3', 10, channels=('a', 'b'))
    assert x3.shape == (10,)
    assert x3.dtype == np.dtype([('a', float), ('b', float)])
    assert x3.itemtype == float
    assert bool(x3.channels)

    x4 = Signal('x4', 10, dtype=int, channels=('a', 'b'))
    assert x4.shape == (10,)
    assert x4.dtype == np.dtype([('a', int), ('b', int)])
    assert x4.itemtype == int
    assert bool(x4.channels)

    x5 = Signal('x5', 10, buffer=np.arange(20))
    assert x5.shape == (10,)
    assert x5.dtype == int
    assert x5.itemtype == int
    assert not bool(x5.channels)
    assert all(x5 == np.arange(10))

    x6 = Signal('x6', 10, buffer=np.arange(20), channels=('a', 'b'))
    assert x6.shape == (10,)
    assert x6.dtype == np.dtype([('a', int), ('b', int)])
    assert x6.itemtype == int
    assert bool(x6.channels)
    assert all(tuple(c1) == tuple(c2) for c1, c2 in
               zip(x6, np.arange(20).reshape(10, 2)))

    x7 = np.arange(10).view(Signal)
    assert x7.shape == (10,)
    assert x7.dtype == int
    assert x7.itemtype == int
    assert not bool(x7.channels)
    assert all(x7 == np.arange(10))

    x8 = np.arange(20).view(dtype=([('a', int), ('b', int)])).view(Signal)
    assert x8.shape == (10,)
    assert x8.dtype == np.dtype([('a', int), ('b', int)])
    assert x8.itemtype == int
    assert bool(x8.channels)
    assert all(tuple(c1) == tuple(c2) for c1, c2 in
               zip(x8, np.arange(20).reshape(10, 2)))
