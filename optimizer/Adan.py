"""Adan optimizer implementation."""

import re

import tensorflow as tf

from keras.optimizers.optimizer_experimental import optimizer


class Adan(optimizer.Optimizer):
    r"""Optimizer that implements the Adan algorithm.
    Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models
    https://arxiv.org/abs/2208.06677
    """

    def __init__(
            self,
            learning_rate=0.001,
            weight_decay=0.05,
            beta_1=0.98,
            beta_2=0.92,
            beta_3=0.99,
            epsilon=1e-16,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="Adan",
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
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.epsilon = epsilon
        if self.weight_decay is None:
            raise ValueError(
                "Missing value of `weight_decay` which is required and"
                " must be a float value.")

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._beliefs = []
        self._prev_gradients = []
        self._velocities = []
        for var in var_list:
            self._beliefs.append(self.add_variable_from_reference(model_variable=var, variable_name="v"))
            self._momentums.append(self.add_variable_from_reference(model_variable=var, variable_name="m"))
            self._prev_gradients.append(self.add_variable_from_reference(model_variable=var, variable_name="p"))
            self._velocities.append(self.add_variable_from_reference(model_variable=var, variable_name="n"))

    def _use_weight_decay(self, variable):
        exclude_from_weight_decay = getattr(self, "_exclude_from_weight_decay", [])
        exclude_from_weight_decay_names = getattr(self, "_exclude_from_weight_decay_names", [])
        if variable in exclude_from_weight_decay:
            return False
        for name in exclude_from_weight_decay_names:
            if re.search(name, variable.name) is not None:
                return False
        return True

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        var_dtype = variable.dtype
        lr = tf.cast(self.learning_rate, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, var_dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, var_dtype), local_step)
        beta_3_power = tf.pow(tf.cast(self.beta_3, var_dtype), local_step)
        alpha_n = tf.sqrt(1.0 - beta_3_power)
        alpha_m = alpha_n / (1.0 - beta_1_power)
        alpha_v = alpha_n / (1.0 - beta_2_power)
        index = self._index_dict[self._var_key(variable)]
        m = self._momentums[index]
        v = self._beliefs[index]
        p = self._prev_gradients[index]
        n = self._velocities[index]
        one_minus_beta_1 = (1 - self.beta_1)
        one_minus_beta_2 = (1 - self.beta_2)
        one_minus_beta_3 = (1 - self.beta_3)

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.scatter_add(tf.IndexedSlices((gradient.values - m) * one_minus_beta_1, gradient.indices))
            diff = (gradient.values - p) * tf.cast(local_step != 1.0, var_dtype)
            v.scatter_add(tf.IndexedSlices((diff - v) * one_minus_beta_2), gradient.indices)
            n.scatter_add(tf.IndexedSlices(
                (tf.math.square(gradient.values + one_minus_beta_2 * diff) - n) * one_minus_beta_3,
                gradient.indices))
            p.scatter_update(tf.IndexedSlices(gradient.values, gradient.indices))
        else:
            # Dense gradients.
            m.assign_add((gradient - m) * one_minus_beta_1)
            diff = (gradient - p) * tf.cast(local_step != 1.0, var_dtype)
            v.assign_add((diff - v) * one_minus_beta_2)
            n.assign_add((tf.math.square(gradient + one_minus_beta_2 * diff) - n) * one_minus_beta_3)
            p.assign(gradient)
        var_t = tf.math.rsqrt(n + self.epsilon) * (alpha_m * m + one_minus_beta_2 * v * alpha_v)
        # Apply step weight decay
        if self._use_weight_decay(variable):
            wd = tf.cast(self.weight_decay, variable.dtype)
            var_t = var_t + variable * wd
        variable.assign_sub(var_t * lr)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "weight_decay": self.weight_decay,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "beta_3": self.beta_3,
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


Adan.__doc__ = Adan.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
