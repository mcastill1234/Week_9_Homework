from util import *


class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        """s:       the current state
           x:       the given input
           returns: the next state"""
        raise NotImplementedError

    def output_fn(self, s):
        """s:       the current state
           returns: the corresponding output"""
        raise NotImplementedError

    def transduce(self, input_seq):
        """input_seq: the given list of inputs
           returns:   list of outputs given the inputs"""
        state = self.start_state
        output = []
        for inp in input_seq:
            state = self.transition_fn(state, inp)
            output.append(self.output_fn(state))
        return output


class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s


class Binary_Addition(SM):
    start_state = (0, 0)  # Change

    def transition_fn(self, s, x):
        (carry, digit) = s
        (i0, i1) = x
        total = i0 + i1 + carry
        return 1 if total > 1 else 0, total % 2

    def output_fn(self, s):
        (carry, digit) = s
        return digit


class Reverser(SM):
    start_state = ([], 'input')

    def transition_fn(self, s, x):
        (symbols, mode) = s
        if x == 'end':
            return symbols, 'output'
        elif mode == 'input':
            return symbols + [x], mode
        else:
            return symbols[:-1], mode

    def output_fn(self, s):
        (symbols, mode) = s
        if mode == 'output' and len(symbols) > 0:
            return symbols[-1]
        else:
            return None


class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        self.Wsx = Wsx
        self.Wss = Wss
        self.Wo = Wo
        self.Wss_0 = Wss_0
        self.Wo_0 = Wo_0
        self.l = self.Wsx.shape[1]
        self.m = self.Wss.shape[1]
        self.n = self.Wo.shape[1]
        self.start_state = np.zeros((self.n, 1))
        self.f1 = f1
        self.f2 = f2

    def transition_fn(self, s, x):
        var1 = self.Wsx @ x
        var2 = self.Wss @ s
        var3 = self.f1(var1 + var2 + self.Wss_0)
        return var3

    def output_fn(self, s):
        var_o1 = self.Wo @ s + self.Wo_0
        var_o2 = self.f2(var_o1)
        return var_o2
