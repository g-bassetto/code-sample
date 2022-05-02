# Test data access and entry names parsing.

import numpy as np
from unittest import TestCase
from maprfutils.data.input import Input
from maprfutils.data.block import Block
from maprfutils.data.utils import *


class TestUtils(TestCase):

    def test_decode_frames(self):
        self.assertEqual(decode_frames('f1234', pads=2), slice(12, 34))
        self.assertEqual(decode_frames(''), EmptySlice)

    def test_encode_frames(self):
        self.assertEqual(encode_frames(EmptySlice), '')
        self.assertEqual(encode_frames(slice(12, 34), pads=2), 'f1234')

    def test_decode_trials(self):
        self.assertEquals(decode_trials('t30'), (1, 2, 3, 4))
        self.assertEquals(decode_trials(''), EmptySlice)

    def test_encode_trials(self):
        self.assertEquals(encode_trials(EmptySlice), '')
        self.assertEquals(encode_trials((1, 2, 3, 4), pads=2), 't30')


class TestBlock(TestCase):

    def setUp(self):
        self.b1 = Block(2, 2)

    def test_empty_ctor(self):
        self.assertEquals(str(self.b1), '')

    def test_frames_setter1(self):
        self.b1.frames = slice(12, 34)
        self.assertEquals(str(self.b1), 'f1234')

    def test_frames_setter2(self):
        self.b1.enc_frames = 'f1234'
        self.assertEquals(self.b1.frames, slice(12, 34))

    def test_trials_setter1(self):
        self.b1.trials = (1, 2, 3, 4)
        self.assertEquals(str(self.b1), 't30')

    def test_trials_setter2(self):
        self.b1.enc_trials = 't30'
        self.assertEquals(self.b1.trials, (1, 2, 3, 4))

    def test_to_string(self):
        self.b1.frames = slice(12, 34)
        self.b1.trials = (1, 2, 3, 4)
        self.assertEquals(str(self.b1), 'f1234t30')

    def test_repr(self):
        b = Block(2, 2)
        b.frames = slice(12, 34)
        b.trials = (1, 2, 3, 4)
        self.assertEquals(b, eval(repr(b)))

    def test_parse(self):
        b1 = Block.parse('f1234t30', 2, 2)
        b2 = Block.parse(str(b1), 2, 2)
        self.assertEquals(b2.frames, b1.frames)
        self.assertEquals(b2.trials, b1.trials)
        b3 = Block.parse(str(b1), 2, 2)
        self.assertEquals(b3.frames, b1.frames)
        self.assertEquals(b3.trials, b1.trials)

    def test_comparisons(self):
        b1 = Block.parse('f1234t30', 2, 2)
        b2 = Block.parse('f1234t30', 2, 4)
        b3 = Block.parse('f1234t30', 2, 2)
        self.assertNotEqual(b1, b2)
        self.assertEquals(b1, b3)
        self.assert_(b1.equiv(b2))
        self.assert_(b2.equiv(b1))


class TestInput(TestCase):

    def setUp(self):
        self.inp = inp = Input()
        inp.frames = np.arange(36).reshape([4, 3, 3])
        inp.spikes = np.arange(8).reshape(2, 4).T + 1

    def test_eq_same_size(self):
        inp2 = Input()
        inp2.frames = np.arange(36).reshape([4, 3, 3])
        inp2.spikes = np.arange(8).reshape(2, 4).T + 1
        self.assertEqual(self.inp, inp2)
        self.assertEqual(inp2, self.inp)

    def test_ne_same_size(self):
        inp2 = Input()
        inp2.frames = 0 * np.arange(36).reshape([4, 3, 3])
        inp2.spikes = 0 * np.arange(8).reshape(2, 4).T + 1
        self.assertNotEqual(self.inp, inp2)
        self.assertNotEqual(inp2, self.inp)

    def test_ne_diff_size(self):
        inp2 = Input()
        inp2.frames = 0 * np.arange(16).reshape([4, 2, 2])
        inp2.spikes = 0 * np.arange(8).reshape(2, 4).T + 1
        self.assertNotEqual(self.inp, inp2)
        self.assertNotEqual(inp2, self.inp)

    def test_block_select_frames(self):
        result = Block.parse('f24', 1).select_frames(self.inp)
        target = self.inp.frames[2:4]
        self.assertListEqual(result.tolist(), target.tolist())

    def test_block_select_spikes(self):
        result = Block.parse('t2', 1, 1).select_spikes(self.inp)
        target = np.array([5, 6, 7, 8]).reshape(-1, 1)
        self.assertListEqual(result.tolist(), target.tolist())

    def test_select_all_frames(self):
        result = self.inp[Block.parse('t1', 1, 1)]
        self.assertIs(result.parent, self.inp)
        self.assertEqual(len(result.frames), len(self.inp.frames))
        self.assertEqual(len(result.spikes), len(self.inp.spikes))

    def test_select_all_trials(self):
        block = Block.parse('f24', 1, 1)
        result = self.inp[block]
        self.assertIs(result.parent, self.inp)
        self.assertListEqual(result.frames.tolist(),
                             self.inp.frames[2:4].tolist())
        self.assertListEqual(result.spikes.tolist(),
                             self.inp.spikes[2:4].tolist())

    def test_select_all(self):
        result = self.inp[Block.parse('', 1, 1)]
        self.assertIsNone(result.parent)
        self.assertIs(result, self.inp)

    def test_slice_indexing_all(self):
        result = self.inp[:]
        self.assertIsNone(result.parent)
        self.assertIsNot(result, self.inp)
        self.assertEqual(len(result), len(self.inp))
        self.assertListEqual(result.frames.tolist(),
                             self.inp.frames.tolist())
        self.assertListEqual(result.spikes.tolist(),
                             self.inp.spikes.tolist())

    def test_slice_indexing(self):
        result = self.inp[2:]
        self.assertIsNotNone(result.parent)
        self.assertIs(result.parent, self.inp)
        self.assertIsNot(result, self.inp)
        self.assertEqual(len(result), 2)
        self.assertListEqual(result.frames.tolist(),
                             self.inp.frames[2:].tolist())
        self.assertListEqual(result.spikes.tolist(),
                             self.inp.spikes[2:].tolist())

    def test_indexing(self):
        self.assertRaises(TypeError, lambda: self.inp[3])
        self.assertRaises(TypeError, lambda: self.inp[3, :])
        self.assertRaises(TypeError, lambda: self.inp[:, 3])
        self.assertRaises(TypeError, lambda: self.inp[1:3, :])
