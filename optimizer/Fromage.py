"""Fromage optimizer implementation."""

import re

import tensorflow as tf

from keras.optimizers.optimizer_experimental import optimizer


class Fromage(optimizer.Optimizer):
    r"""Optimizer that implements the Fromage algorithm.
    On the distance between two neural networks and the stability of learning
    https://arxiv.org/abs/2002.03432
    https://github.com/deepmind/deepmind-research/blob/master/nfnets/optim.py
    """

    def __init__(
            self,
            learning_rate=0.001,
            weight_decay=0.004,
            momentum=0.9,
            epsilon=1e-5,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="Fromage",
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
        self.epsilon = epsilon

        if self.weight_decay is None:
            raise ValueError(
                "Missing value of `weight_decay` which is required and"
                " must be a float value."
            )

    def build(self, var_list):
        """Initialize optimizer variables. 
        Args:
          var_list: list of model variables to build Fromage variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

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
        mult = tf.math.rsqrt(1.0 + lr ** 2),
        # Apply step weight decay
        if self._use_weight_decay(variable):
            wd = tf.cast(self.weight_decay, variable.dtype)
            variable.assign_sub(variable * wd * lr)
        var_norm = tf.maximum(tf.norm(variable), self.epsilon)
        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            grad_norm = tf.maximum(tf.norm(gradient.values), self.epsilon)
            variable.assign(variable, (variable - lr * gradient.values * (var_norm / grad_norm)) * mult)
        else:
            # Dense gradients.
            grad_norm = tf.maximum(tf.norm(gradient), self.epsilon)
            variable.assign(variable, (variable - lr * gradient * (var_norm / grad_norm)) * mult)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "weight_decay": self.weight_decay,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
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


Fromage.__doc__ = Fromage.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
