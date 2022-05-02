# Tests for the torch-based PoissonGLM implementation
# and the custom Newton optimizer (IRLS)

import numpy as np
import numpy.random as nr
from numpy.testing import assert_allclose, assert_array_equal
from numpy.testing import assert_array_almost_equal
from scipy.optimize import check_grad
from maprfutils.utils.math_utils import approx_fprime

from unittest import TestCase
from maprfutils.distributions import MvNormalTau
from maprfutils.core.nonlin import SoftPlus, Exponential
from maprfutils.models.glm import PoissonGLM
from maprfutils.models import glm
from scipy.optimize import minimize


ATOL_BETA = 1e-4
RTOL_BETA = 1e-6

ATOL_GRAD = 1e-6
RTOL_GRAD = 1e-6

ATOL_HESS = 1e-6
RTOL_HESS = 1e-6

NON_LINEARITY = Exponential()
N_OBS = 20
N_PAR = 5
STEP_SIZE = 0.5


def build_model():
    x = np.random.randn(N_OBS, N_PAR)
    y = np.random.poisson(1, size=(N_OBS, 1))
    model = PoissonGLM(x, y, NON_LINEARITY, delta=STEP_SIZE)
    model.beta.set_prior(MvNormalTau(N_PAR, tau=np.eye(N_PAR)))
    return model


def random_param():
    return np.random.randn(N_PAR)


class TestPoissonGLM(TestCase):

    def setUp(self):
        self.model = build_model()
        self.x = self.model.x
        self.y = self.model.y

    def test_set_y(self):
        y = np.random.poisson(1, size=(N_OBS, 2))
        y1, y2 = y[:, 0], y
        #
        self.assertEqual(y1.ndim, 1)
        self.model.y = y1
        self.assertEqual(self.model.y.shape, (len(y), 1))
        self.assertEqual(self.model.s.shape, (len(y),))
        self.assertTrue(np.alltrue(self.model.s == y1))

        self.model.y = y2
        self.assertEqual(self.model.y.shape, (len(y), 2))
        self.assertEqual(self.model.s.shape, (len(y),))
        self.assertTrue(np.alltrue(self.model.s == y2.sum(axis=1)))

    def test_update_cost(self):
        self.model.beta = random_param()
        self.assertTrue(np.isscalar(self.model.cost))

    def test_update_grad_1(self):
        self.model.y = np.random.poisson(1, size=(N_OBS, 1))
        self.helper_test_grad(self.model)

    def test_update_grad_2(self):
        self.model.y = np.random.poisson(1, size=(N_OBS, 2))
        self.helper_test_grad(self.model)

    def helper_test_grad(self, model):
        def func(x):
            model.beta = x
            return model.cost

        def grad(x):
            self.model.beta = x
            return self.model.grad

        for _ in range(10):
            xi = np.random.randn(5)
            target_grad = approx_fprime(func, xi)
            tested_grad = grad(xi)
            assert_allclose(tested_grad, target_grad,
                            atol=ATOL_GRAD, rtol=RTOL_GRAD)

    def test_update_hessian(self):
        def func(x, index):
            self.model.beta = x
            return self.model.grad[index]

        def grad(x, index):
            self.model.beta = x
            return self.model.hess[index]

        n_params = len(self.model.beta)
        x = np.random.randn(n_params)
        for k in range(n_params):
            target_grad = approx_fprime(func, x, k)
            tested_grad = grad(x, k)
            assert_allclose(tested_grad, target_grad,
                            atol=ATOL_HESS, rtol=RTOL_HESS)

    def test_optimize(self):
        self.model.beta = np.zeros_like(self.model.beta)
        self.model.optimize()
        control_logp = self.model.logp
        control_beta = np.array(self.model.beta, copy=True)

        optimizer = glm.NewtonOptimizer()
        optimizer.optimize(self.model, np.zeros_like(self.model.beta))
        tested_logp = self.model.logp
        tested_beta = np.array(self.model.beta, copy=True)

        assert_allclose(tested_beta, control_beta,
                        atol=ATOL_BETA, rtol=RTOL_BETA)
        self.assertAlmostEqual(tested_logp, control_logp)


class TestOptimizers(TestCase):
    def setUp(self) -> None:
        self.model = build_model()

        def func(β):
            self.model.beta = β
            return self.model.cost

        def grad(β):
            self.model.beta = β
            return self.model.grad

        p0 = np.zeros_like(self.model.beta)
        self.target_results = minimize(func, p0)
        self.target_results_w_grad = minimize(func, p0, jac=grad)

    def test_scipy_optimizer_params(self):
        control = self.target_results.x
        result = self.target_results_w_grad.x
        assert_allclose(result, control, atol=ATOL_BETA, rtol=RTOL_BETA)

    def test_scipy_optimizer_cost(self):
        control = self.target_results.fun
        result = self.target_results_w_grad.fun
        self.assertAlmostEqual(result, control)

    def test_newton_optimizer_params(self):
        optimizer = glm.NewtonOptimizer()
        optimizer.optimize(self.model, np.zeros_like(self.model.beta))
        control = self.target_results.x
        result = self.target_results_w_grad.x
        assert_allclose(result, control, atol=ATOL_BETA, rtol=RTOL_BETA)

    def test_newton_optimizer_cost(self):
        optimizer = glm.NewtonOptimizer()
        optimizer.optimize(self.model, np.zeros_like(self.model.beta))
        control = self.target_results.fun
        result = self.target_results_w_grad.fun
        self.assertAlmostEqual(result, control)
