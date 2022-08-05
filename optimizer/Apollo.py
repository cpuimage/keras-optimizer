"""Apollo optimizer implementation."""

import re

import tensorflow as tf

from keras.optimizers.optimizer_experimental import optimizer


class Apollo(optimizer.Optimizer):
    r"""Optimizer that implements the Apollo algorithm.
        Apollo: An Adaptive Parameter-wise Diagonal Quasi-Newton Method for Nonconvex Stochastic Optimization
        https://arxiv.org/abs/2009.13586
        https://github.com/XuezheMax/apollo
    """

    def __init__(
            self,
            learning_rate=0.001,
            weight_decay=4e-5,
            beta_1=0.9,
            rebound=0.01,
            epsilon=1e-7,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="Apollo",
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
        self._learning_rate = self._build_learning_rate(learning_rate * rebound)
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.rebound = rebound
        self.epsilon = epsilon
        if self.weight_decay is None:
            raise ValueError(
                "Missing value of `weight_decay` which is required and"
                " must be a float value.")

    def build(self, var_list):
        """Initialize optimizer variables.


        Args:
          var_list: list of model variables to build Apollo variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._approx_hessian = []
        self._updates = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(model_variable=var, variable_name="m"))
            self._approx_hessian.append(
                self.add_variable_from_reference(model_variable=var, variable_name="approx_hessian"))
            self._updates.append(
                self.add_variable_from_reference(model_variable=var, variable_name="update"))

    def _use_weight_decay(self, variable):
        exclude_from_weight_decay = getattr(
            self, "_exclude_from_weight_decay", [])
        exclude_from_weight_decay_names = getattr(
            self, "_exclude_from_weight_decay_names", [])
        if variable in exclude_from_weight_decay:
            return False
        for name in exclude_from_weight_decay_names:
            if re.search(name, variable.name) is not None:
                return False
        return True

    def _norm(self, x, power=4.0):
        pow_x = tf.pow(tf.abs(x), power)
        return tf.maximum(tf.pow(tf.reduce_sum(pow_x), (1.0 / power)), self.epsilon)

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1 = tf.cast(self.beta_1, variable.dtype)
        rebound = tf.cast(self.rebound, variable.dtype)
        alpha = (1.0 - beta_1) / (1. - tf.pow(beta_1, local_step))
        index = self._index_dict[self._var_key(variable)]
        m = self._momentums[index]
        approx_hessian = self._approx_hessian[index]
        update = self._updates[index]
        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            delta_grad = gradient.values - m
            m.scatter_add(tf.IndexedSlices(delta_grad * alpha), gradient.indices)
            v = tf.math.divide_no_nan(update, tf.square(self._norm(update)))
            v_sq = update * v
            approx_hessian.scatter_sub(tf.IndexedSlices(
                v_sq * alpha * tf.reduce_sum(delta_grad * v) + tf.reduce_sum(approx_hessian * v_sq)), gradient.indices)
            var_t = m / tf.maximum(tf.abs(approx_hessian), rebound)
            # Apply step weight decay
            if self._use_weight_decay(variable):
                wd = tf.cast(self.weight_decay, variable.dtype)
                var_t = var_t + variable * wd
            update.scatter_update(tf.IndexedSlices(var_t, gradient.indices))
        else:
            # Dense gradients.
            delta_grad = gradient - m
            m.assign_add(delta_grad * alpha)
            v = tf.math.divide_no_nan(update, tf.square(self._norm(update)))
            v_sq = update * v
            approx_hessian.assign_sub(
                v_sq * alpha * tf.reduce_sum(delta_grad * v) + tf.reduce_sum(approx_hessian * v_sq))
            var_t = m / tf.maximum(tf.abs(approx_hessian), rebound)
            # Apply step weight decay
            if self._use_weight_decay(variable):
                wd = tf.cast(self.weight_decay, variable.dtype)
                var_t = var_t + variable * wd
            update.assign(var_t)
        variable.assign_sub(lr * update)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "weight_decay": self.weight_decay,
                "beta_1": self.beta_1,
                "rebound": self.rebound,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
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


Apollo.__doc__ = Apollo.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
