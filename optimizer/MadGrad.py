"""MadGrad optimizer implementation."""

import re

import tensorflow as tf

from keras.optimizers.optimizer_experimental import optimizer


class MadGrad(optimizer.Optimizer):
    r"""Optimizer that implements the MadGrad algorithm.
    Adaptivity without Compromise: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic Optimization
    https://arxiv.org/abs/2101.11075
    """

    def __init__(
            self,
            learning_rate=0.001,
            weight_decay=0.004,
            momentum=0.9,
            power=1.0 / 3.0,
            epsilon=1e-6,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="MadGrad",
            **kwargs
    ):
        super().__init__(
            name=name,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.power = power
        self.epsilon = epsilon

        if self.weight_decay is None:
            raise ValueError(
                "Missing value of `weight_decay` which is required and"
                " must be a float value."
            )

    def build(self, var_list):
        """Initialize optimizer variables.

        Args:
          var_list: list of model variables to build MadGrad variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._grad_sum_sq = []
        self._s = []
        self._x0 = []
        for var in var_list:
            self._grad_sum_sq.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="grad_sum_sq"))
            self._s.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="s"))
            self._x0.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="x0", initial_value=var))

    def _use_weight_decay(self, variable):
        exclude_from_weight_decay = getattr(
            self, "_exclude_from_weight_decay", []
        )
        exclude_from_weight_decay_names = getattr(
            self, "_exclude_from_weight_decay_names", []
        )
        if variable in exclude_from_weight_decay:
            return False
        for name in exclude_from_weight_decay_names:
            if re.search(name, variable.name) is not None:
                return False
        return True

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        power = tf.cast(self.power, variable.dtype)
        index = self._index_dict[self._var_key(variable)]
        grad_sum_sq = self._grad_sum_sq[index]
        s = self._s[index]
        x0 = self._x0[index]
        # Apply step weight decay
        if self._use_weight_decay(variable):
            wd = tf.cast(self.weight_decay, variable.dtype)
            variable.assign_sub(variable * wd * lr)

        if isinstance(gradient, tf.IndexedSlices):
            sk_grad = lr * tf.sqrt(local_step) * gradient.values
            # Sparse gradients.
            s.scatter_add(tf.IndexedSlices(sk_grad, gradient.indices))
            grad_sum_sq.scatter_add(tf.IndexedSlices(sk_grad * gradient.values, gradient.indices))
        else:
            # Dense gradients.
            sk_grad = lr * tf.sqrt(local_step) * gradient
            s.assign_add(sk_grad)
            grad_sum_sq.assign_add(sk_grad * gradient)
        rms = tf.maximum(tf.pow(grad_sum_sq, power), self.epsilon)
        z = x0 - (s / rms)
        var_t = (1 - self.momentum) * variable + self.momentum * z
        variable.assign(var_t)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "weight_decay": self.weight_decay,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "power": self.power,
            }
        )
        return config

    def exclude_from_weight_decay(self, var_list=None, var_names=None):
        """Exclude variables from weight decays.

        This method must be called before the optimizer's `build` method is
        called. You can set specific variables to exclude out, or set a list of
        strings as the anchor words, if any of which appear in a variable's
        name, then the variable is excluded.

        Args:
            var_list: A list of `tf.Variable`s to exclude from weight decay.
            var_names: A list of strings. If any string in `var_names` appear
                in the model variable's name, then this model variable is
                excluded from weight decay. For example, `var_names=['bias']`
                excludes all bias variables from weight decay.
        """
        if hasattr(self, "_built") and self._built:
            raise ValueError(
                "`exclude_from_weight_decay()` can only be configued before "
                "the optimizer is built."
            )

        self._exclude_from_weight_decay = var_list or []
        self._exclude_from_weight_decay_names = var_names or []


MadGrad.__doc__ = MadGrad.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
