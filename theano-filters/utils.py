import numpy as np
import numpy.linalg as nla

import theano
import theano.tensor as tt
import scipy.special as special
from collections import namedtuple

from theano import config
from theano import shared as tshared

import GPy as gpy

floatx = getattr(np, config.floatX)

from collections import OrderedDict


class UpdatesDict(OrderedDict):

    def __setitem__(self, key, value):
        super().__setitem__(key, tt.cast(value, key.dtype))


def make_buffer_for(var, dtype=None):
    if isinstance(var, list):
        varlist = var
        return [make_buffer_for(v) for v in varlist]
    else:
        if dtype is None:
            dtype = var.dtype
        if var.ndim == 0:
            # dealing with a scalar
            val = np.cast[dtype](0)
            return theano.shared(val, var.name)
        else:
            arr = np.array([], dtype, ndmin=var.ndim)
            return theano.shared(arr, var.name)


class FiniteDiff:

    def __init__(self, fn, dx=None):
        self.fn = fn
        self.dx = dx

    def __call__(self, x):
        npars = x.shape[-1]
        dx = np.diag(self.dx)

        xp = x[:, None, :] + dx / 2
        yp = self.fn(inputs=xp.reshape(-1, npars))

        xm = x[:, None, :] - dx / 2
        ym = self.fn(inputs=xm.reshape(-1, npars))

        dy = np.reshape(yp - ym, (-1, npars))
        return dy / self.dx


def default_kernel(ptype, bounds=None):
    import GPy as gpy
    k1 = gpy.kern.Bias(len(ptype))
    k2 = gpy.kern.Matern52(len(ptype), ARD=True)
    return k1 + k2


class Surrogate:

    @property
    def ni(self):
        return len(self.itype)

    def __init__(self, itype, kernel=None, bounds=None):
        self.itype = itype
        if kernel is None:
            kernel = default_kernel(self.itype)
        self.kernel = kernel

    def gradient(self, *args, inputs=None):
        if args:
            inputs = np.array(list(zip(*args)), dtype=self.itype)
        outer_shape = inputs.shape
        inputs = inputs.view('f8').reshape(*outer_shape, -1)

        x = inputs.view('f8').reshape(-1, self.ni)
        g, _ = self.gp.predictive_gradients(x)
        return g

    def __call__(self, *args, inputs=None, std=False):
        if args:
            inputs = np.array(list(zip(*args)), dtype=self.itype)
        outer_shape = inputs.shape if inputs.dtype == self.itype else inputs.shape[
                                                                      :-1]
        inputs = inputs.view('f8').reshape(*outer_shape, -1)

        assert inputs.shape[-1] == self.ni
        x = inputs.view('f8').reshape(-1, self.ni)
        y, s = self.gp.predict_noiseless(x)

        if not std:
            return np.reshape(y, outer_shape)
        else:
            return np.reshape(y, outer_shape), np.reshape(s, outer_shape)

    def train(self, inputs, output, optimize=True):
        inputs = inputs.view('f8').reshape(-1, self.ni)
        output = output.reshape(-1, 1)

        if hasattr(self, 'gp'):
            self.gp.set_XY(inputs, output)
        else:
            self.gp = gpy.models.GPRegression(
                inputs, output, kernel=self.kernel,
                noise_var=1e-6, normalizer=False
            )
            self.gp.Gaussian_noise.variance.fix()
        if optimize:
            self.gp.optimize()


class DataSetter:
    def __init__(self, buffers):
        self.buffers = buffers
        x = buffers['x'].type()
        y = buffers['y'].type()
        self._update = theano.function(
            [x, y], allow_input_downcast=True,
            updates=[
                (buffers['x'], x),
                (buffers['y'], y),
            ]
        )

    def __call__(self, x, y, return_values=False):
        self._update(x, y)
        if return_values:
            return x, y


class DataGetter:
    def __init__(self, buffers):
        self.buffers = buffers

    def __call__(self):
        x = self.buffers['x'].get_value()
        y = self.buffers['y'].get_value()
        return x, y


def mkshared(ndim, dtype=None, name=None):
    if dtype is None:
        dtype = floatx
    value = np.array([], dtype=dtype, ndmin=ndim)
    return theano.shared(value, name, allow_downcast=True)


def parse(value):
    if np.issubsctype(np.array(value), float):
        return floatx(value)
    else:
        return value


class usegrid:

    def __init__(self, grid):
        self.grid = grid

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            maker = func(*args, **kwargs)
            return maker[self.grid]

        return wrapper


class ChoiceParam:

    def __init__(self, choices, default=None):
        self.names = dict()
        self.choices = choices
        if default is None:
            default = choices[0]
        self.default = default

    def __set_name__(self, owner, name):
        self.names[owner] = '_' + name

    def __get__(self, inst, owner):
        if inst is None:
            return self
        attrname = self.names[owner]
        return getattr(inst, attrname, self.default)

    def __set__(self, inst, value):
        attrname = self.names[type(inst)]
        setattr(inst, attrname, value)


class Gaussian2D:

    def __init__(self, scale=1.0, normalize=True):
        self.scale = scale
        self.normalize = normalize

    def __getitem__(self, grid):
        pts = np.stack(grid)
        sq_rad = np.sum(pts ** 2, axis=0)
        value = np.exp(-sq_rad / (2 * self.scale ** 2))
        if self.normalize:
            const = 2 * np.pi * self.scale ** 2
            value = value / const
        return value


class CenterSurround:
    mode = ChoiceParam(['on', 'off'])

    def __init__(self, w_cen, w_sur, ratio=1.0, mode='on'):
        if w_cen > w_sur:
            raise ValueError()
        self.cen = Gaussian2D(w_cen)
        self.sur = Gaussian2D(w_sur)

        self.ratio = ratio
        self.mode = mode

    @property
    def sign(self):
        return 1 if self.mode == 'on' else -1

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        if value > 2 or value < 0:
            raise ValueError()
        self._ratio = value

    @property
    def w_cen(self):
        return self.cen.scale

    @property
    def w_sur(self):
        return self.sur.scale

    def __getitem__(self, grid):
        cen = self.cen[grid]  # * (1 + self.ratio) / 2
        sur = self.sur[grid]  # * (1 - self.ratio) / 2
        return self.sign * (cen - sur * self.ratio)


def logistic(a=0., b=1.):
    a = tt.as_tensor_variable(a, 'a')
    b = tt.as_tensor_variable(b, 'a')

    def apply(x):
        z_pos = 1 / (1 + tt.exp(-x))
        z_neg = tt.exp(x) / (1 + tt.exp(x))
        return a + (b - a) * tt.where(x > 0, z_pos, z_neg)

    return apply


def invlogistic(a=0., b=1.):
    a = tt.as_tensor_variable(a, 'a')
    b = tt.as_tensor_variable(b, 'a')

    def apply(x):
        return tt.log((x - a) / (b - x))

    return apply


logit = invlogistic
expit = tt.nnet.sigmoid


def stable_log(x):
    return np.log(x + 1e-20)


def stable_exp(x):
    return np.exp(x) - 1e-20


def raised_cos(s, phi, dphi):
    u = (s - phi) * np.pi / (2 * dphi)
    u[u > +np.pi] = +np.pi
    u[u < -np.pi] = -np.pi
    return 0.5 * (1 + np.cos(u))


class RaisedCosines:
    bases = None

    @property
    def range_y(self):
        x_min, x_max = self.range_x
        y_min = stable_log(x_min + self.offset)
        y_max = stable_log(x_max + self.offset)
        return y_min, y_max

    @property
    def n_bases(self):
        return len(self.bases)

    def make_times(self, n):
        y_min, y_max = self.range_y
        delta_y = (y_max - y_min) / (n - 1)
        t_max = -self.offset + stable_exp(y_max + 2 * delta_y)
        return np.arange(0, t_max, self.binsize)

    def __init__(self, lims, offset=0, binsize=1.0):
        self.offset = offset
        self.range_x = lims
        self.binsize = binsize

    def build(self, n, normalize=False):
        y_min, y_max = self.range_y
        delta_y = (y_max - y_min) / (n - 1)
        centers = np.linspace(y_min, y_max, n)

        times = self.make_times(n)
        u = stable_log(times + self.offset)
        bases = [raised_cos(u, to, delta_y) / 2 for to in centers]
        #
        if normalize:
            # TODO: implement ortho-normalization
            raise NotImplementedError()
            # bases = np.stack(bases, axis=1)
            # self.bases = la.svd(bases, full_matrices=False)[0]
        else:
            self.bases = bases
        self.times = times

        return self

    def apply(self, signal):
        from scipy.signal import lfilter

        def conv(b):
            return lfilter(b, np.ones(1), signal, axis=0)

        channels = [conv(b) for b in self.bases]
        return np.stack(channels, axis=1)


def slice_time_series(x, lags):
    n = x.shape[0]
    padding = np.zeros((lags - 1,) + x[0].shape)
    new_x = np.concatenate([x, padding], axis=0)
    indices = np.arange(n).reshape(n, 1) - np.arange(lags)
    return new_x[indices]


def apply_basis_function(x, b):
    shp = x.shape[1:]
    n = x.shape[0]
    m = b.shape[0]
    p = b.shape[1]
    x = slice_time_series(x, m).reshape(n, m, -1)
    y = np.dot(b.T, x)
    return y.reshape(n, p, *shp)


def create_raised_cosines(nh, lims, b, step, normalize=False):
    def rcos(s, phi, dphi):
        u = (s - phi) * np.pi / (2 * dphi)
        u[u > +np.pi] = +np.pi
        u[u < -np.pi] = -np.pi
        return 0.5 * (1 + np.cos(u))

    def log(x):
        return np.log(x + 1e-20)

    def exp(x):
        return np.exp(x) - 1e-20

    x_min, x_max = lims
    y_min = log(x_min + b)
    y_max = log(x_max + b)
    dc = (y_max - y_min) / (nh - 1)
    xo = np.linspace(y_min, y_max, nh)
    t_max = -b + exp(y_max + 2 * dc)
    t = np.arange(0, t_max, step)
    u = log(t + b)
    k = np.stack([rcos(u, to, dc) for to in xo]).T / 2
    if not normalize:
        return t, k
    else:
        return t, nla.svd(k, full_matrices=False)[0]


def empty(ndim, dtype=None):
    if dtype is None:
        dtype = config.floatX
    if ndim == 0:
        ctor = getattr(np, dtype)
        return ctor(0)
    else:
        return np.array([], dtype=dtype, ndmin=ndim)


def pol2cart(v):
    return v[0] * tt.stack([tt.cos(v[1]), tt.sin(v[1])])


def cart2pol(v):
    u1 = tt.sqrt(v[0] ** 2 + v[1] ** 2)
    u2 = tt.arctan2(v[1], v[0])
    return tt.stack([u1, u2])


NormalParams = namedtuple('NormalParams', ['mu', 'sd'])


def normal_from_ci(p1, p2, f=None):
    β, α = [x for x in zip(p1, p2)]
    if f is not None:
        try:
            T = getattr(np, f)
        except AttributeError:
            T = getattr(special, f)
        β = [T(x) for x in β]
    ζ = [special.erfinv(2 * x - 1) for x in α]

    den = ζ[1] - ζ[0]
    σ = np.sqrt(.5) * (β[1] - β[0]) / den
    # μ = (β[1] * ζ[1] - β[0] * ζ[0]) / den
    μ = 0.5 * (β[1] + β[0]) + np.sqrt(0.5) * σ * (ζ[0] + ζ[1])

    return NormalParams(mu=μ, sd=σ)


def mkbuffer(*value, size=None, ndim=None, **kwargs):
    if value:
        value, = value
        if isinstance(value, str):
            if value == 'scalar':
                return mkbuffer(0.0, **kwargs)
            elif value == 'vector':
                return mkbuffer(ndim=1, **kwargs)
            elif value == 'matrix':
                return mkbuffer(ndim=2, **kwargs)
        else:
            return tshared(value, **kwargs)
    else:
        if size:
            value = np.zeros(size, dtype=config.floatX)
        else:
            value = np.array([], dtype=config.floatX, ndmin=ndim)
        return tshared(value, **kwargs)


def slice_signal(x, lags, *, reverse=True):
    bins = x.shape[0]
    padding = np.zeros(shape=(lags-1,) + x.shape[1:], dtype=x.dtype)
    offset = np.arange(bins).reshape(-1, 1)
    shifts = np.arange(lags).reshape(1, -1)
    if reverse:
        ind = offset - shifts + (lags - 1)
    else:
        ind = offset + shifts
    return np.concatenate([padding, x], axis=0)[ind]


def array(data, ndim=None):
    return np.array(data, dtype=config.floatX, ndmin=ndim)


class ContextMeta(type):

    def get_context(cls):
        try:
            return cls.__context__[-1]
        except IndexError:
            return None

    def set_context(cls, ctx):
        return cls.__context__.append(ctx)

    def pop_context(cls):
        cls.__context__.pop()

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if not hasattr(cls, '__context__'):
            cls.__context__ = list()

    def __call__(cls, *args, **kwargs):
        inst = super().__call__(*args, **kwargs)
        ctx = cls.get_context()
        if ctx:
            ctx.append(inst)
        return inst


class AbstractCtx(metaclass=ContextMeta):

    def __enter__(self):
        type(self).set_context(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        type(self).pop_context()


class FunctionsChain(AbstractCtx):

    @property
    def count(self):
        return len(self.stages)

    def __init__(self, *stages):
        self.stages = list(stages)

    def __getitem__(self, index):
        return self.__class__(*self.stages[index])

    def __call__(self, x):
        if self.count == 0:
            return x
        else:
            f = self.stages[-1]
            z = self[:-1](x)
            return f(z)

    def __enter__(self):
        type(self).set_context(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        type(self).pop_context()

    def append(self, stage):
        self.stages.append(stage)
