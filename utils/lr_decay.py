# https://github.com/Ctiely/A2C/blob/master/utils/lr_decay.py
# https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py


def constant(p):
    return 1


def linear(p):
    return 1 - p


def middle_drop(p):
    eps = 0.75
    if 1 - p < eps:
        return eps * 0.1
    return 1 - p


def double_linear_con(p):
    p *= 2
    eps = 0.125
    if 1 - p < eps:
        return eps
    return 1 - p


def double_middle_drop(p):
    eps1 = 0.75
    eps2 = 0.25
    if 1 - p < eps1:
        if 1 - p < eps2:
            return eps2 * 0.5
        return eps1 * 0.1
    return 1 - p


class LearningRateDecay(object):
    def __init__(self, v, nvalues, lr_decay_method):
        """
        :param lr_decay_method: {'linear', 'constant', 'double_linear_con', 'middle_drop', 'double_middle_drop'}
        """
        self.n = 0.
        self.v = v
        self.nvalues = nvalues

        lr_decay_methods = {
            'linear': linear,
            'constant': constant,
            'double_linear_con': double_linear_con,
            'middle_drop': middle_drop,
            'double_middle_drop': double_middle_drop
        }

        self.decay = lr_decay_methods[lr_decay_method]

    def value(self):
        current_value = self.v * self.decay(self.n / self.nvalues)
        self.n += 1.
        return current_value

    def get_value_for_steps(self, steps):
        return self.v * self.decay(steps / self.nvalues)

if __name__ == "__main__":
    lr_decay = LearningRateDecay(v=1.0, nvalues=100000, lr_decay_method="linear")
    for _ in range(10):
        print(lr_decay.value())