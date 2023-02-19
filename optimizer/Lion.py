"""Lion optimizer implementation."""
import tensorflow as tf

from keras.optimizers.optimizer_experimental import optimizer


class Lion(optimizer.Optimizer):
    r"""Optimizer that implements the Lion algorithm.
    https://github.com/google/automl/tree/master/lion
    """

    def __init__(
            self,
            learning_rate: float = 0.0001,
            beta_1: float = 0.9,
            beta_2: float = 0.99,
            weight_decay: float = 0.0,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema: bool = False,
            ema_momentum: float = 0.99,
            ema_overwrite_frequency=None,
            jit_compile: bool = True,
            name="Lion",
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
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._m = []
        for var in var_list:
            self._m.append(self.add_variable_from_reference(model_variable=var, variable_name="m"))

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        var_dtype = variable.dtype
        wd = tf.cast(self.weight_decay, var_dtype)
        lr = tf.cast(self.learning_rate, var_dtype)
        beta_1 = tf.cast(self.beta_1, var_dtype)
        beta_2 = tf.cast(self.beta_2, var_dtype)
        one_minus_beta_1 = 1 - beta_1
        one_minus_beta_2 = 1 - beta_2
        index = self._index_dict[self._var_key(variable)]
        m = self._m[index]
        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            variable.scatter_sub(
                tf.IndexedSlices(lr * (tf.math.sign(m * beta_1 + gradient.values * one_minus_beta_1) + variable * wd),
                                 gradient.indices))
            m.scatter_update(tf.IndexedSlices(m * beta_2 + gradient.values * one_minus_beta_2, gradient.indices))
        else:
            variable.assign_sub(lr * (tf.math.sign(m * beta_1 + gradient * one_minus_beta_1) + variable * wd))
            m.assign(m * beta_2 + gradient * one_minus_beta_2)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "weight_decay": self.weight_decay,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
            }
        )
        return config


Lion.__doc__ = Lion.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
