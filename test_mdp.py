from unittest import TestCase
from mdp import TabularQ, value, greedy
from util import *


class Test(TestCase):
    def test_value(self):
        q = TabularQ([0, 1, 2, 3], ['b', 'c'])
        q.set(0, 'b', 5)
        q.set(0, 'c', 10)
        self.assertEqual(10, value(q, 0))

    def test_greedy(self):
        q = TabularQ([0, 1, 2, 3], ['b', 'c'])
        q.set(0, 'b', 5)
        q.set(0, 'c', 10)
        q.set(1, 'b', 2)
        self.assertEqual('c', greedy(q, 0))
        self.assertEqual('b', greedy(q, 1))
