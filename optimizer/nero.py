"""nero optimizer implementation."""

import re

import tensorflow as tf

from keras.optimizers.optimizer_experimental import optimizer


class nero(optimizer.Optimizer):
    r"""Optimizer that implements the nero algorithm.
        Learning by Turning: Neural Architecture Aware Optimisation
        https://arxiv.org/abs/2102.07227
        https://github.com/jxbz/nero
    """

    def __init__(
            self,
            learning_rate=0.001,
            weight_decay=0.004,
            beta_1=0.9,
            epsilon=1e-16,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="nero",
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
        self.epsilon = epsilon

        if self.weight_decay is None:
            raise ValueError(
                "Missing value of `weight_decay` which is required and"
                " must be a float value."
            )

    def euclidean_norm(self, x, axis=1, keepdims=True):
        square_sum = tf.reduce_sum(tf.square(x), axis=axis, keepdims=keepdims)
        return tf.sqrt(square_sum)

    def neuron_update(self, x):
        x = x - self.neuron_mean(x)
        return tf.math.divide_no_nan(x, self.neuron_norm(x))

    def neuron_scale(self, x):
        x = tf.reduce_mean(self.neuron_norm(x))
        return tf.where(x == 0.0, 0.01, x)

    def neuron_norm(self, x):
        ndims = x.shape.ndims
        if ndims > 1:
            shape = tf.shape(x)
            view_shape = [shape[0]] + [1] * (ndims - 1)
            x = tf.reshape(x, (shape[0], -1))
            return tf.reshape(self.euclidean_norm(x, axis=1, keepdims=True), view_shape)
        else:
            return tf.abs(x)

    def neuron_mean(self, x):
        ndims = x.shape.ndims
        if ndims > 1:
            shape = tf.shape(x)
            view_shape = [shape[0]] + [1] * (ndims - 1)
            x = tf.reshape(x, (shape[0], -1))
            return tf.reshape(tf.reduce_mean(x, axis=1, keepdims=True), view_shape)
        else:
            raise Exception("neuron_mean not defined on 1D tensors.")

    def build(self, var_list):
        """Initialize optimizer variables.

        Args:
          var_list: list of model variables to build nero variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._velocities = []
        self._scales = []
        for var in var_list:
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
            self._scales.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="s", initial_value=self.neuron_scale(var)
                )
            )

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
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        index = self._index_dict[self._var_key(variable)]
        s = self._scales[index]
        v = self._velocities[index]
        alpha = s * lr * tf.math.rsqrt(1 - beta_1_power)
        # Apply step weight decay
        if self._use_weight_decay(variable):
            wd = tf.cast(self.weight_decay, variable.dtype)
            variable.assign_sub(variable * wd * lr)

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            v.assign_add(-v * (1 - self.beta_1))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(self.neuron_norm(gradient.values)) * (1 - self.beta_1) + self.epsilon,
                    gradient.indices))
            grad_normed = gradient.values * alpha * tf.math.rsqrt(v)
        else:
            # Dense gradients.
            v.assign_add((tf.square(self.neuron_norm(gradient)) - v) * (1 - self.beta_1) + self.epsilon)
            grad_normed = gradient * alpha * tf.math.rsqrt(v)
        if variable.shape.ndims > 1:
            variable.assign(self.neuron_update(
                tf.cond(local_step == 1.0, lambda: self.neuron_update(variable), lambda: variable) - grad_normed))
        else:
            variable.assign_sub(grad_normed)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "weight_decay": self.weight_decay,
                "beta_1": self.beta_1,
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


nero.__doc__ = nero.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
