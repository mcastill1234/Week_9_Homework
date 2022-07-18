from unittest import TestCase
from sm import *


def softmax(z):
    v = np.exp(z)
    return v / np.sum(v, axis=0)


class Test(TestCase):

    def test_accumulator(self):
        res = Accumulator().transduce([-1, 2, 3, -2, 5, 6])
        self.assertEqual(res, [-1, 1, 4, 2, 7, 13])

    def test_binary_addition(self):
        res = Binary_Addition().transduce([(1, 1), (1, 0), (0, 0)])
        self.assertEqual([0, 0, 1], res)

    def test_reverser(self):
        res = Reverser().transduce(['foo', ' ', 'bar'] + ['end'] + list(range(5)))
        self.assertEqual([None, None, None, 'bar', ' ', 'foo', None, None, None], res)

    def test_rnn(self):
        Wsx1 = np.array([[0.1],
                         [0.3],
                         [0.5]])
        Wss1 = np.array([[0.1, 0.2, 0.3],
                         [0.4, 0.5, 0.6],
                         [0.7, 0.8, 0.9]])
        Wo1 = np.array([[0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6]])
        Wss1_0 = np.array([[0.01],
                           [0.02],
                           [0.03]])
        Wo1_0 = np.array([[0.1],
                          [0.2]])
        in1 = [np.array([[0.1]]),
               np.array([[0.3]]),
               np.array([[0.5]])]

        rnn = RNN(Wsx1, Wss1, Wo1, Wss1_0, Wo1_0, np.tanh, softmax)
        expected = np.array([[[0.4638293846951024], [0.5361706153048975]],
                             [[0.4333239107898491], [0.566676089210151]],
                             [[0.3821688606165438], [0.6178311393834561]]])
        self.assertTrue(np.allclose(expected, rnn.transduce(in1)))

    def test_arnn(self):
        Wsx2 = np.array([[1, 0, 0]]).T
        Wss2 = np.array([[0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0]])
        Wo2 = np.array([[1, -2, 3]])
        Wss2_0 = np.array([[0, 0, 0]]).T
        Wo2_0 = np.array([[0]])
        in2 = np.array([[5.0]])

        f1 = lambda x: x
        f2 = lambda x: x

        rnn = RNN(Wsx2, Wss2, Wo2, Wss2_0, Wo2_0, f1, f2)
        print(rnn.transduce(in2))


