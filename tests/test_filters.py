# Test torch-based linear filter implementation.

import torch
import numpy as np
import scipy.signal as ss

from unittest import TestCase
from maprfutils.filters import MultiLinear, SOSMatrix, SOSFilter


class TestMultiLinear(TestCase):
    device = None

    def setUp(self) -> None:
        self.signal = torch.ones(9, 3, 5)
        self.kernel = torch.ones(3, 5, 2)

    def test_build(self):
        f = MultiLinear(self.signal, self.kernel, device=self.device)
        self.assertEqual(f.kernel.device, torch.device(self.device))
        self.assertEqual(f.signal.device, torch.device(self.device))

    def test_kernel_set(self):
        f = MultiLinear(signal=self.signal, device=self.device)
        with self.assertRaises(AssertionError):
            f.kernel = torch.ones(3, 4, 2)

    def test_signal_set(self):
        f = MultiLinear(kernel=self.kernel, device=self.device)
        f.signal = self.signal
        self.assertIsNotNone(getattr(f, 'kernel', None))
        f.signal = torch.ones(9, 3, 4)
        self.assertIsNone(getattr(f, 'kernel', None))

    def test_set_device(self):
        f = MultiLinear(self.signal, self.kernel)
        f.set_device(self.device)
        self.assertEqual(f.signal.device, self.device)
        self.assertEqual(f.kernel.device, self.device)

    def test_call(self):
        f = MultiLinear(kernel=self.kernel, device=self.device)
        r = f(self.signal)
        expected = torch.full((9, 2), 15, device=self.device)
        self.assertTrue(torch.allclose(r, expected))
        self.assertEqual(r.device, self.device)

    def test_apply(self):
        f = MultiLinear(signal=self.signal, device=self.device)
        r = f.apply(self.kernel)
        expected = torch.full((9, 2), 15, device=self.device)
        self.assertTrue(torch.allclose(r, expected))
        self.assertEqual(r.device, self.device)


class TestMultiLinearCPU(TestMultiLinear):

    def setUp(self) -> None:
        super().setUp()
        self.device = torch.device('cpu')


class TestMultiLinearGPU(TestMultiLinear):

    def setUp(self) -> None:
        super().setUp()
        self.device = torch.device('cuda:0')


del TestMultiLinear


class TestSOSMatrix(TestCase):
    def setUp(self) -> None:
        self.b1 = np.array([1, -1, 0])
        self.a1 = np.array([1, 1, 0.25])
        self.b2 = np.array([1, 1, 0])
        self.a2 = np.array([1, -1, 0.25])
        self.sosmat = np.array([[self.b1, self.a1], [self.b2, self.a2]]) \
            .reshape(-1, 6)
        self.b, self.a = ss.sos2tf(self.sosmat)

    def test_create(self):
        with self.assertRaises(ValueError):
            SOSMatrix.empty(0.5)
            SOSMatrix.empty(-5)

        m = SOSMatrix.empty(3)
        self.assertTrue(np.isnan(m).all())

    def test_len_sections(self):
        m = SOSMatrix.from_tf(self.b, self.a)
        self.assertEqual(m.n_sections, 2)
        self.assertEqual(m[0].n_sections, 1)

    def test_from_tf_vector(self):
        m = SOSMatrix.from_tf(self.b1, self.a1)
        self.assertEqual(m.shape, (1, 6))
        self.assertTrue(np.allclose(m.b, self.b1))
        self.assertTrue(np.allclose(m.a, self.a1))

    def test_from_tf_matrix(self):
        b = np.stack([self.b1, self.b2])
        a = np.stack([self.a1, self.a2])
        m = SOSMatrix.from_tf(b, a)
        self.assertEqual(m.shape, (2, 6))
        b, a = ss.sos2tf(m)
        self.assertTrue(np.allclose(self.b, b))
        self.assertTrue(np.allclose(self.a, a))

    def test_from_tf(self):
        m = SOSMatrix.from_tf(self.b, self.a)
        b, a = ss.sos2tf(m)
        self.assertEqual(m.shape, (2, 6))
        self.assertTrue(np.allclose(b, self.b))
        self.assertTrue(np.allclose(a, self.a))

    def test_getter_1_sec(self):
        m = SOSMatrix.from_tf(self.b1, self.a1)
        self.assertTrue((m.b == self.b1).all())
        self.assertTrue((m.a == self.a1).all())

    def test_getter_n_sec(self):
        m = SOSMatrix.from_tf(self.b, self.a)
        self.assertEqual(len(m[0].b), 3)
        self.assertEqual(len(m[0].a), 3)
        self.assertTrue(np.allclose(m.b, self.b))
        self.assertTrue(np.allclose(m.a, self.a))

    def test_b_setter(self):
        m = SOSMatrix.from_tf(self.b, self.a)
        m[0].b = self.b1
        with self.assertRaises(TypeError):
            m.b = self.b

    def test_a_setter(self):
        m = SOSMatrix.from_tf(self.b, self.a)
        m[0].a = self.a1

        b = m[0].b.copy()
        m[0].a = 2 * self.a1
        self.assertTrue(np.allclose(m[0].b, b / 2))

        with self.assertRaises(TypeError):
            m.b = self.b

    def test_discrete(self):
        dt = 0.1
        b, a = ss.bilinear(self.b, self.a, 1 / dt)
        md1 = SOSMatrix.from_tf(b, a)
        md2 = SOSMatrix.from_tf(self.b, self.a).discrete(1 / dt)

        self.assertTrue(np.allclose(md1.a, md2.a))
        self.assertTrue(np.allclose(md1.b, md2.b))


class TestSOSFilter(TestCase):

    def setUp(self) -> None:
        self.matrix = np.array([
            [[1, -1, 0], [1, 1, 0.25]],
            [[1, 1, 0], [1, -1, 0.25]],
        ]).reshape(-1, 6)
        self.signal = torch.ones(20)

    def test_signal_setter(self):
        f = SOSFilter()
        f.signal = self.signal
        self.assertEqual(f.signal.device, torch.device('cpu'))
        self.assertEqual(f.output.device, torch.device('cpu'))
        self.assertEqual(f.output.shape, self.signal.shape)

    def test_signal_setter_gpu(self):
        f = SOSFilter()
        gpu_signal = self.signal.to('cuda:0')
        f.signal = gpu_signal
        self.assertEqual(f.signal.device, torch.device('cpu'))
        self.assertEqual(f._signal.device, torch.device('cuda:0'))
        self.assertEqual(f.output.device, torch.device('cpu'))

        self.assertEqual(f.output.shape, self.signal.shape)

        gpu_signal[0] = 2
        self.assertEqual(f.signal[0], 2)

    def test_matrix_setter(self):
        f = SOSFilter()
        f.matrix = self.matrix
        self.assertEqual(f.matrix.device, torch.device('cpu'))

    def test_apply(self):
        f = SOSFilter()

        with self.assertRaises(AttributeError):
            f.apply(self.matrix)

        f.signal = self.signal
        r = f.apply(self.matrix)
        target = ss.sosfilt(self.matrix, self.signal, axis=0)
        self.assertTrue(np.allclose(r, target))
        self.assertEqual(r.device, torch.device('cpu'))

    def test_apply_gpu(self):
        signal = self.signal.cuda()
        f = SOSFilter(signal=signal)

        r = f.apply(self.matrix)
        target = ss.sosfilt(self.matrix, 1 * self.signal, axis=0)
        self.assertTrue(np.allclose(r, target))

        signal *= 2
        r = f.apply(self.matrix)
        target = 2 * ss.sosfilt(self.matrix, 1 * self.signal, axis=0)
        self.assertTrue(np.allclose(r, target))
        self.assertEqual(r.device, torch.device('cpu'))

