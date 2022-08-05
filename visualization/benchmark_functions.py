import math

import tensorflow as tf  # TF2


# https://en.wikipedia.org/wiki/Test_functions_for_optimization
class Benchmark:
    def __init__(self):
        super().__init__()
        # Domain should be a list of lists where each list yields the minimum and maximum value for one of the variables.
        pass

    def calculate(self, x_var, y_var):
        pass

    def clip_to_domain(self, x_var, y_var):
        if x_var < self.domain[0][0]:
            x_var = tf.Variable(self.domain[0][0])
        if x_var > self.domain[0][1]:
            x_var = tf.Variable(self.domain[0][1])

        if y_var < self.domain[1][0]:
            y_var = tf.Variable(self.domain[1][0])
        if y_var > self.domain[1][1]:
            y_var = tf.Variable(self.domain[1][1])

        return x_var, y_var


class beale(Benchmark):
    def __init__(self):
        super().__init__()

        self.domain = [
            [-4.5, 4.5],
            [-4.5, 4.5],
        ]

    def calculate(self, x_var, y_var):
        x_var = tf.Variable(0.0) if x_var is None else x_var
        y_var = tf.Variable(0.0) if y_var is None else y_var

        return tf.square(1.5 - x_var + x_var * y_var) + tf.square(2.25 - x_var + x_var * tf.square(y_var)) + tf.square(
            2.625 - x_var + x_var * tf.pow(y_var, 3))


class rosenbrock(Benchmark):
    def __init__(self):
        super().__init__()

        self.domain = [
            [-1.5, 1.5],
            [-1.5, 1.5],
        ]

    def calculate(self, x_var, y_var):
        x_var = tf.Variable(0.0) if x_var is None else x_var
        y_var = tf.Variable(0.0) if y_var is None else y_var

        return (1.0 - x_var) ** 2.0 + 100.0 * (y_var - x_var ** 2.0) ** 2.0


class sphere(Benchmark):
    def __init__(self):
        super().__init__()

        self.domain = [
            [-1.5, 1.5],
            [-1.5, 1.5],
        ]

    def calculate(self, x, y):
        # noisy hills of the cost function
        def __f1(x, y):
            return -1 * tf.sin(x * x) * tf.cos(3 * y * y) * tf.exp(-(x * y) * (x * y)) - tf.exp(-(x + y) * (x + y))

        # bivar gaussian hills of the cost function
        def __f2(x, y, x_mean, y_mean, x_sig, y_sig):
            normalizing = 1 / (2 * math.pi * x_sig * y_sig)
            x_exp = (-1 * tf.square(x - x_mean)) / (2 * tf.square(x_sig))
            y_exp = (-1 * tf.square(y - y_mean)) / (2 * tf.square(y_sig))
            return normalizing * tf.exp(x_exp + y_exp)

        x = tf.Variable(0.0) if x is None else x
        y = tf.Variable(0.0) if y is None else y

        # two local minima near (0, 0)
        #     z = __f1(x, y)

        # 3rd local minimum at (-0.5, -0.8)
        z = -1 * __f2(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.35, y_sig=0.35)

        # one steep gaussian trench at (0, 0)
        #     z -= __f2(x, y, x_mean=0, y_mean=0, x_sig=0.2, y_sig=0.2)

        # three steep gaussian trenches
        z -= __f2(x, y, x_mean=1.0, y_mean=-0.5, x_sig=0.2, y_sig=0.2)
        z -= __f2(x, y, x_mean=-1.0, y_mean=0.5, x_sig=0.2, y_sig=0.2)
        z -= __f2(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.2, y_sig=0.2)

        return z
