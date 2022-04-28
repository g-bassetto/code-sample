from numpy import array
from theano.tensor.sharedvar import TensorSharedVariable
from ..signal import channel_parser


class PipelineState(TensorSharedVariable):
    """
    PipelineState is a wrapper around TensorSharedVariable used to store a
    symbolic reference to a particular state of a Pipeline.
    """
    _channels = ()

    @property
    def channels(self) -> tuple:
        """
        The channels of the underlying signal.
        """
        return self._channels

    @channels.setter
    def channels(self, value: tuple) -> None:
        """
        Set the channels of the underlying signal.
        :var value: a list of channel names
        """
        assert self.get_value().shape[-1] == len(value)
        self._channels = value

    @property
    def nchannels(self) -> int:
        """
        Number of channels of the underlying signal.
        """
        return len(self.channels) if self.channels else 1


def empty(ndims: int, dtype: str=None, channels=None, name: str=None) -> \
        PipelineState:
    """
    Create an empty PipelineState with the specified attributes
    :var ndims: number of dimensions
    :var dtype: float format
    :var channels: a list of channel names
    :var name: a name to associate to the state
    :return: an empty s
    """

    from theano.tensor import TensorType
    from theano import config

    if dtype is None:
        dtype = config.floatX
    if channels is None:
        channels = ()
    channels = channel_parser(channels)
    nc = len(channels) if channels else 1

    value = array([0] * nc, dtype=dtype, ndmin=ndims + 1)
    ttype = TensorType(dtype, [False] * value.ndim)
    var = PipelineState(name, ttype, value, False, True)
    if channels:
        var.channels = channels

    return var


def as_pipeline_state(var: TensorSharedVariable, channels=None) -> PipelineState:
    """
    Wraps PipelineState around a theano shared variable.
    :var var: the variable to wrap
    :var channels: channels of the signal
    :return: a PipelineState object wrapping the variable.
    """
    if not isinstance(var, PipelineState):
        var.__class__ = PipelineState
        if channels:
            var.channels = channels
    return var
