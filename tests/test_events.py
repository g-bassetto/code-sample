# Tests for the Event class.

from unittest import TestCase
from maprfutils.events import Event

def foo(sender):
    pass


def bar(sender):
    pass


def baz(sender):
    pass


class TestEvent(TestCase):

    def test_add_observer(self):
        e = Event()
        e.add_observer(foo)
        e.add_observer(bar)
        self.assertListEqual(e.observers, [foo, bar])

    def test_add_duplicate_observer(self):
        e = Event()
        e.add_observer(foo)
        e.add_observer(bar)
        e.add_observer(foo)
        self.assertListEqual(e.observers, [foo, bar])

    def test_del_observer(self):
        e = Event()
        e.add_observer(foo)
        e.add_observer(bar)
        self.assertListEqual(e.observers, [foo, bar])
        e.del_observer(foo)
        self.assertListEqual(e.observers, [bar])

    def test_del_disowned_observer(self):
        e = Event()
        e.add_observer(foo)
        e.add_observer(bar)
        self.assertListEqual(e.observers, [foo, bar])
        e.del_observer(baz)
        self.assertListEqual(e.observers, [foo, bar])

