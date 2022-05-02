# Test parametric kernel shapes.

import numpy as np

from unittest import TestCase

from maprfutils.shapes import ShapeFeatures
from maprfutils.shapes import Param, ParamsList, SOSMatrix
from maprfutils import shapes
from maprfutils.utils import parse_weights_grid_xy


class TestParamsList(TestCase):

    def setUp(self) -> None:
        self.p1 = p1 = Param('param1', 1.0)
        self.p2 = p2 = Param('param2', 2.0)
        self.p3 = p3 = Param('param3', 3.0)
        self.pl = ParamsList([p1, p2])
        self.ps = [p1, p2, p3]

    def test_init(self):
        self.assertIs(getattr(self.pl, 'param1', None), self.p1)
        self.assertIs(getattr(self.pl, 'param2', None), self.p2)
        self.assertIsNone(getattr(self.pl, 'param3', None))

    def test_append(self):
        self.pl.append(self.p3)
        self.assertListEqual(self.pl, self.ps)

    def test_remove(self):
        self.pl.remove(self.p2)
        self.assertIsNone(getattr(self.pl, 'param2', None))
        self.assertNotIn(self.p2, self.pl)

    def test_pop(self):
        p2 = self.pl.pop(1)
        self.assertIs(p2, self.p2)
        self.assertIsNone(getattr(self.pl, 'param2', None))
        self.assertNotIn(self.p2, self.pl)


class TestGamma(TestCase):
    def setUp(self) -> None:
        self.shape = shapes.Gamma(5)

    def test_call_kernel(self):
        t = np.arange(20)
        s = self.shape(t)
        self.assertEqual(s.value.shape, t.shape)
        self.assertEqual(s.update_value, s.update_kernel)
        self.assertIs(getattr(s, 'axis_t', None), t)

    def test_call_matrix(self):
        dt = 0.1
        s = self.shape(dt=dt)
        self.assertIsInstance(s.value, SOSMatrix)
        self.assertEqual(s.value.shape, (3, 6))
        self.assertEqual(s.update_value, s.update_matrix)
        self.assertIs(getattr(s, 't_step', None), dt)


class TestGammaDiff(TestCase):
    def setUp(self) -> None:
        self.shape = shapes.GammaDiff(5)
        self.c1 = shapes.Gamma(5)
        self.c2 = shapes.Gamma(7)

    def test_scale_changed(self):
        new_value = 2
        self.shape.scale = new_value
        self.assertEqual(self.shape.component1.scale, new_value)
        self.assertEqual(self.shape.component2.scale, new_value)

    def test_kernel_value(self):
        t = np.arange(20)
        s = self.shape(t)
        target = self.c1.kernel(t) - self.c2.kernel(t)
        self.assertTrue(np.allclose(s.value, target))

    def test_call_kernel(self):
        t = np.arange(20)
        s = self.shape(t)
        self.assertEqual(s.value.shape, t.shape)
        self.assertEqual(s.update_value, s.update_kernel)
        self.assertIs(getattr(s, 'axis_t', None), t)

    def test_matrix(self):
        dt = 0.1

        self.shape.kappa = 0
        target = self.c1.matrix(dt)
        m = self.shape.matrix(dt)
        self.assertTrue(np.allclose(target, m))

        self.shape.kappa = 0.5
        ta = self.c2.matrix()
        ta[0].b = [1, 2, 0.5]
        td = ta.discrete(1 / dt)
        ma = self.shape.matrix(dt=None)
        md = self.shape.matrix(dt=dt)
        self.assertTrue(np.allclose(ta, ma))
        self.assertTrue(np.allclose(td, md))

    def test_call_matrix(self):
        dt = 0.1
        s = self.shape(dt=dt)
        self.assertIsInstance(s.value, SOSMatrix)
        self.assertEqual(s.value.shape, (4, 6))
        self.assertEqual(s.update_value, s.update_matrix)
        self.assertEqual(getattr(s, 't_step', None), dt)


class TestGaborMode(TestCase):

    def test_func(self):
        factory = shapes.GaborMode
        self.assertIs(factory.cos.func, np.cos)
        self.assertIs(factory.sin.func, np.sin)

    def test_parse(self):
        factory = shapes.GaborMode
        self.assertIs(factory.parse('cos'), factory.cos)
        self.assertIs(factory.parse('sin'), factory.sin)


class TestGaborFeatureFlags(TestCase):
    def setUp(self) -> None:
        x = np.linspace(-2, 2, 41)
        y = np.linspace(2, -2, 41)
        grids = np.meshgrid(x, y)

        shape = shapes.Gabor('cos')
        shape.phase = 0.5
        shape.scale_x = 0.5
        shape.scale_y = 0.5

        self.gabor = shape(*grids)
        self.gabor.features += ShapeFeatures.SUM_TO_ZERO
        self.gabor.features += ShapeFeatures.NORMALIZED

    def test_analytic_bias_off(self):
        # analytic solution off
        self.gabor.features -= ~ShapeFeatures.USE_ANALYTIC_BIAS
        self.gabor._update_offset()
        target = shapes.numeric_gabor_offset(self.gabor)
        self.assertEqual(self.gabor.offset, target)

    def test_analytic_bias_on(self):
        # analytic solution on
        self.gabor.features += ShapeFeatures.USE_ANALYTIC_BIAS
        self.gabor._update_offset()
        target = shapes.gabor_offset(self.gabor)
        self.assertAlmostEqual(self.gabor.offset, target)

    def test_analytic_norm_off(self):
        # analytic solution off
        self.gabor.features &= ~ShapeFeatures.USE_ANALYTIC_NORM
        self.gabor.parameters_changed()
        self.assertIsNone(getattr(self.gabor, 'energy', None))

    def test_analytic_solution_on(self):
        # analytic solution on
        self.gabor.features |= ShapeFeatures.USE_ANALYTIC_NORM
        self.gabor.parameters_changed()
        self.assertIsNotNone(getattr(self.gabor, 'energy', None))


class TestGaborOffset(TestCase):

    def setUp(self) -> None:
        x = np.linspace(-5, 5, 101)
        y = np.linspace(5, -5, 101)
        self.grids = np.meshgrid(x, y)

    def test_cos(self):
        shape = shapes.Gabor('cos')
        self.evaluate(shape)

    def test_sin(self):
        shape = shapes.Gabor('sin')
        self.evaluate(shape)

    def evaluate(self, shape):
        shape.frequency = 1.0
        shape.scale_x = 0.5
        shape.scale_y = 0.5

        phases = np.random.uniform(-np.pi, np.pi, 10)
        for phi in phases:
            shape.phase = phi
            shape = shape(*self.grids)
            # numeric solution
            n_sol = shapes.numeric_gabor_offset(shape)
            # analytic solution
            a_sol = shapes.gabor_offset(shape)

            self.assertAlmostEqual(float(n_sol), float(a_sol))


class TestGaboVolume(TestCase):

    def setUp(self) -> None:
        x = np.linspace(-5, 5, 101)
        y = np.linspace(5, -5, 101)
        self.grids = np.meshgrid(x, y)

    def test_cos(self):
        shape = shapes.Gabor('cos')
        self.evaluate(shape)

    def test_sin(self):
        shape = shapes.Gabor('sin')
        self.evaluate(shape)

    def evaluate(self, shape):
        shape.frequency = 1.0
        shape.scale_x = 0.5
        shape.scale_y = 0.5

        phases = np.random.uniform(-np.pi, np.pi, 10)
        for phi in phases:
            shape.phase = phi
            shape = shape(*self.grids)
            # numeric solution
            n_sol = shapes.numeric_gabor_volume(shape)
            # analytic solution
            a_sol = shapes.gabor_volume(shape)

            self.assertAlmostEqual(float(n_sol), float(a_sol))


class TestGaborRemoveBias(TestCase):
    def setUp(self) -> None:
        self.shape = shapes.Gabor('cos')
        self.shape.angle = 0.5
        self.shape.phase = 0.5
        self.shape.scale_x = 0.5
        self.shape.scale_y = 0.5
        self.shape.features += ShapeFeatures.SUM_TO_ZERO

        self.grid_shape = (101, 101)
        x = np.linspace(-5, 5, self.grid_shape[1])
        y = np.linspace(5, -5, self.grid_shape[0])
        self.grids = np.meshgrid(x, y)

    def test_remove_bias_off(self):
        self.shape.features = ShapeFeatures.NONE
        gabor = self.shape(*self.grids)
        self.assertEqual((gabor.weights * gabor.value).sum(),
                         shapes.numeric_gabor_volume(gabor))

    def test_remove_bias_a(self):
        self.shape.features += ShapeFeatures.USE_ANALYTIC_BIAS
        gabor = self.shape(*self.grids)
        self.assertAlmostEqual((gabor.weights * gabor.value).sum(), 0)

    def test_remove_bias_n(self):
        self.shape.features -= ShapeFeatures.USE_ANALYTIC_BIAS
        gabor = self.shape(*self.grids)
        self.assertAlmostEqual((gabor.weights * gabor.value).sum(), 0)


class TestGabor(TestCase):
    def setUp(self) -> None:
        self.shape = shapes.Gabor(self.mode.name)
        self.shape.angle = 0.5
        self.shape.phase = 0.5
        self.shape.scale_x = 0.5
        self.shape.scale_y = 0.5

        self.grid_shape = (41, 41)
        x = np.linspace(-2, 2, self.grid_shape[1])
        y = np.linspace(2, -2, self.grid_shape[0])
        self.grids = np.meshgrid(x, y)
        self.grid_x = self.grids[0]
        self.grid_y = self.grids[1]

    def test_init(self):
        self.assertIs(self.shape.mode, self.mode)

    def test_grid_x(self):
        x = self.shape.grid_x
        self.assertEqual(len(x), 0)
        self.shape.grid_x = self.grid_x
        self.assertEqual(self.shape.grid_x.shape, self.grid_shape)
        self.assertTrue(np.all(self.shape.grid_x == self.grid_x))

    def test_grid_y(self):
        y = self.shape.grid_y
        self.assertEqual(len(y), 0)
        self.shape.grid_y = self.grid_y
        self.assertEqual(self.shape.grid_y.shape, self.grid_shape)
        self.assertTrue(np.all(self.shape.grid_y == self.grid_y))

    def test_default_parameters(self):
        p = shapes.Gabor.default_parameters()
        self.assertIsInstance(p, shapes.GaborParams)

    def test_value_shape(self):
        value = self.shape(*self.grids).value
        self.assertEqual(value.shape, self.grid_x.shape)

    def test_call(self):
        k = self.shape(*self.grids).value
        w = self.shape.weights
        self.assertEqual(k.shape, self.grid_shape)
        self.assertEqual(w.shape, self.grid_shape)

    def test_normalized_analytic_norm(self):
        self.shape.features += ShapeFeatures.NORMALIZED
        self.shape.features += ShapeFeatures.USE_ANALYTIC_NORM
        self.assertIn(ShapeFeatures.USE_ANALYTIC_NORM, self.shape.features)
        gabor = self.shape(*self.grids)
        self.assertAlmostEqual((gabor.weights * (gabor.value ** 2)).sum(), 1)

    def test_normalized_numeric_norm(self):
        self.shape.features += ShapeFeatures.NORMALIZED
        self.shape.features -= ShapeFeatures.USE_ANALYTIC_NORM
        self.assertNotIn(ShapeFeatures.USE_ANALYTIC_NORM, self.shape.features)
        gabor = self.shape(*self.grids)
        self.assertAlmostEqual((gabor.weights * (gabor.value ** 2)).sum(), 1)


class TestGaborSin(TestGabor):

    def setUp(self) -> None:
        self.mode = shapes.GaborMode.sin
        super().setUp()


class TestGaborCos(TestGabor):

    def setUp(self) -> None:
        self.mode = shapes.GaborMode.cos
        super().setUp()


del TestGabor


class TestShapeFeatures(TestCase):

    def setUp(self) -> None:
        self.flag = ShapeFeatures.NONE

    def test_iadd(self):
        self.flag += ShapeFeatures.USE_ANALYTIC_NORM
        self.assertIn(ShapeFeatures.USE_ANALYTIC_NORM, self.flag)

    def test_isub(self):
        self.assertNotIn(ShapeFeatures.USE_ANALYTIC_BIAS, self.flag)
        self.flag -= ShapeFeatures.USE_ANALYTIC_BIAS
        self.assertNotIn(ShapeFeatures.USE_ANALYTIC_BIAS, self.flag)

        self.flag += ShapeFeatures.USE_ANALYTIC_NORM
        self.flag += ShapeFeatures.USE_ANALYTIC_BIAS
        self.assertIn(ShapeFeatures.USE_ANALYTIC_NORM, self.flag)
        self.assertIn(ShapeFeatures.USE_ANALYTIC_BIAS, self.flag)

        self.flag -= ShapeFeatures.USE_ANALYTIC_NORM
        self.assertNotIn(ShapeFeatures.USE_ANALYTIC_NORM, self.flag)
        self.assertIn(ShapeFeatures.USE_ANALYTIC_BIAS, self.flag)

    def test_set(self):
        flag = self.flag.set(ShapeFeatures.USE_ANALYTIC_NORM)
        self.assertIn(ShapeFeatures.USE_ANALYTIC_NORM, flag)

    def test_remove(self):
        flag = self.flag
        flag = flag.set(ShapeFeatures.USE_ANALYTIC_NORM)
        flag = flag.set(ShapeFeatures.USE_ANALYTIC_BIAS)

        self.assertIn(ShapeFeatures.USE_ANALYTIC_NORM, flag)
        self.assertIn(ShapeFeatures.USE_ANALYTIC_BIAS, flag)

        flag = flag.remove(ShapeFeatures.USE_ANALYTIC_NORM)
        self.assertNotIn(ShapeFeatures.USE_ANALYTIC_NORM, flag)
        self.assertIn(ShapeFeatures.USE_ANALYTIC_BIAS, flag)


class TestGaborPair(TestCase):
    def setUp(self) -> None:
        self.shape = shapes.GaborPair()
        self.shape.phase = 0.5
        self.shape.scale_x = 0.5
        self.shape.scale_y = 0.5

        x = np.linspace(-2, 2, 41)
        y = np.linspace(2, -2, 41)
        self.grids = np.meshgrid(x, y)
        self.grid_x = self.grids[0]
        self.grid_y = self.grids[1]

    def test_init(self):
        shape = shapes.GaborPair()
        self.assertNotIn(shape.phase, shape.parameters)
        self.assertEqual(shape.channel1.phase, np.pi / 4)
        self.assertEqual(shape.channel2.phase, np.pi / 4)
        self.assertListEqual(shape.channel1.parameters, [])
        self.assertListEqual(shape.channel2.parameters, [])

    def test_grid_x(self):
        self.shape.grid_x = self.grid_x
        self.assertEqual(self.shape.grid_x.shape, (41, 41))
        self.assertIs(self.shape.channel1.grid_x, self.grid_x)
        self.assertIs(self.shape.channel2.grid_x, self.grid_x)

    def test_grid_y(self):
        self.shape.grid_y = self.grid_y
        self.assertEqual(self.shape.grid_y.shape, (41, 41))
        self.assertIs(self.shape.channel1.grid_y, self.grid_y)
        self.assertIs(self.shape.channel2.grid_y, self.grid_y)

    def test_features(self):
        shape = self.shape
        shape.features += ShapeFeatures.SUM_TO_ZERO
        self.assertIn(ShapeFeatures.SUM_TO_ZERO, shape.channel1.features)
        self.assertIn(ShapeFeatures.SUM_TO_ZERO, shape.channel2.features)

