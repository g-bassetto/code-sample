# Test automatic updates on parameter changes.

import torch
import numpy
import logging
from unittest import TestCase
from maprfutils.operations import Operation, filtercall, accept_if_self, accept_if_parent
from maprfutils.core import ObservableParams

# logging.basicConfig(level=logging.DEBUG)

SETTING_INPUTS_HANDLED = '__setting_inputs_called__'
SETTING_OUTPUT_HANDLED = '__setting_output_called__'
INPUTS_CHANGED_HANDLED = '__inputs_changed_called__'
OUTPUT_CHANGED_HANDLED = '__output_changed_called__'


class DummyOp(Operation):

    def setup(self):
        self.params = ObservableParams(self.__name__ + 'param')
        self.inputs_changed += self.record_inputs_changed
        self.output_changed += self.record_output_changed

    def __call__(self, signal):
        return 2 * signal

    @filtercall(accept_if_self)
    def record_inputs_changed(self, sender, signal):
        setattr(self, INPUTS_CHANGED_HANDLED, sender is self)

    @filtercall(accept_if_self)
    def record_output_changed(self, sender, signal):
        logging.debug(f"  > {self}, {sender}, {self.parent}:"
                      f" {self.parent is sender}")
        setattr(self, OUTPUT_CHANGED_HANDLED, sender is self)

    @filtercall(accept_if_self)
    def on_setting_inputs(self, sender, signal):
        setattr(self, SETTING_INPUTS_HANDLED, sender is self)
        logging.debug(f"  > {self}: signal is of type {type(signal)})\t"
                      f"handled: {getattr(self, SETTING_INPUTS_HANDLED)}")
        self.tag = signal
        # print(f"{self}.on_setting_inputs called by {sender}")

    @filtercall(accept_if_self)
    def on_setting_output(self, sender, signal):
        setattr(self, SETTING_OUTPUT_HANDLED, sender is self)
        logging.debug(f"  > {self}: signal is of type {type(signal)})\t"
                      f"handled: {getattr(self, SETTING_OUTPUT_HANDLED)}")
        # print(f"{self}.on_setting_output called by {sender}")


class TestInitialization(TestCase):

    def setUp(self) -> None:
        self.op = DummyOp('OP')

    def test_name(self):
        self.assertEqual(self.op.__name__, 'OP')

    def test_parent(self):
        self.assertIsNone(self.op.parent)

    def test_inputs_init(self):
        self.assertIsNone(self.op.inputs)

    def test_output_init(self):
        self.assertIsNone(self.op.output)

    def test_class_handlers(self):
        self.assertIn(self.op.on_setting_inputs, Operation.setting_inputs)
        self.assertIn(self.op.on_setting_output, Operation.setting_output)
        self.assertIn(self.op.on_inputs_changed, Operation.inputs_changed)
        self.assertIn(self.op.on_output_changed, Operation.output_changed)

    def test_object_handlers(self):
        self.assertIn(self.op.on_params_changed, self.op.params.changed)


class TestEvents(TestCase):

    def setUp(self):
        self.op1 = DummyOp('Op1')
        self.op2 = DummyOp('Op2')
        self.op2.inputs = numpy.full([2, 2], 3)

    def test_op1_setting_inputs_handled(self):
        self.assertFalse(getattr(self.op1, SETTING_INPUTS_HANDLED, False))

    def test_op2_setting_inputs_handled(self):
        self.assertTrue(getattr(self.op2, SETTING_INPUTS_HANDLED))

    def test_op1_setting_output_handled(self):
        self.assertFalse(getattr(self.op1, SETTING_OUTPUT_HANDLED, False))

    def test_op2_setting_output_handled(self):
        self.assertTrue(getattr(self.op2, SETTING_OUTPUT_HANDLED))

    def test_op1_changed_inputs_handled(self):
        self.assertFalse(getattr(self.op1, INPUTS_CHANGED_HANDLED, False))

    def test_op2_changed_inputs_handled(self):
        self.assertTrue(getattr(self.op2, INPUTS_CHANGED_HANDLED))

    def test_op1_changed_output_handled(self):
        self.assertFalse(getattr(self.op1, OUTPUT_CHANGED_HANDLED, False))

    def test_op2_changed_output_handled(self):
        self.assertTrue(getattr(self.op2, OUTPUT_CHANGED_HANDLED))

    def test_op2_output_value(self):
        target = torch.full([2, 2], 6)
        self.assertListEqual(self.op2.output.tolist(), target.tolist())

    def test_inputs_type_numpy(self):
        self.op1.inputs = numpy.ones([3, 3])
        self.assertIsInstance(self.op1.inputs, torch.Tensor)

    def test_inputs_type_torch(self):
        self.op1.inputs = torch.ones([3, 3])
        self.assertIsInstance(self.op1.inputs, torch.Tensor)


class TestCouplingThenSetting(TestCase):

    def setUp(self):
        self.op1 = DummyOp('Op1')
        self.op2 = DummyOp('Op2')
        self.op2.parent = self.op1
        self.signal = numpy.ones([2, 2])
        self.op1.inputs = self.signal

    def test_io_match(self):
        self.assertIs(self.op1.output, self.op2.inputs)

    def test_op1_inputs_changed(self):
        self.assertTrue(getattr(self.op1, INPUTS_CHANGED_HANDLED))

    def test_op2_inputs_changed(self):
        self.assertTrue(getattr(self.op2, INPUTS_CHANGED_HANDLED))

    def test_output_value(self):
        self.assertListEqual(torch.full([2, 2], 4).tolist(),
                             self.op2.output.tolist())

    def test_op1_setting_inputs(self):
        self.assertIs(self.op1.tag, self.signal)

    def test_op2_setting_output(self):
        self.assertIs(self.op2.tag, self.op1.output)


class TestSettingThenCoupling(TestCase):

    def setUp(self):
        self.signal = numpy.ones([2, 2])
        self.op1 = DummyOp('Op1')
        self.op1.inputs = self.signal
        self.op2 = DummyOp('Op2')
        self.op2.parent = self.op1

    def test_io_match(self):
        self.assertIs(self.op1.output, self.op2.inputs)

    def test_op1_inputs_changed(self):
        self.assertTrue(getattr(self.op1, INPUTS_CHANGED_HANDLED))

    def test_op2_inputs_changed(self):
        self.assertTrue(getattr(self.op2, INPUTS_CHANGED_HANDLED))

    def test_output_value(self):
        self.assertListEqual(torch.full([2, 2], 4).tolist(),
                             self.op2.output.tolist())

    def test_op1_setting_inputs(self):
        self.assertIs(self.op1.tag, self.signal)

    def test_op2_setting_output(self):
        self.assertIs(self.op2.tag, self.op1.output)
