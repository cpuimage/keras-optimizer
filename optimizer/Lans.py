"""lans optimizer implementation."""

import re

import tensorflow as tf

from keras.optimizers.optimizer_experimental import optimizer


class Lans(optimizer.Optimizer):
    """
      References
    - Accelerated Large Batch Optimization of BERT Pretraining in 54 minutes. http://arxiv.org/abs/2006.13484
    - Large Batch Optimization for Deep Learning: Training BERT in 76 minutes. https://arxiv.org/abs/1904.00962v3
    - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. https://arxiv.org/abs/1810.04805
    """

    def __init__(
            self,
            learning_rate=0.001,
            weight_decay=0.004,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="Lans",
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
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        if self.weight_decay is None:
            raise ValueError(
                "Missing value of `weight_decay` which is required and"
                " must be a float value."
            )

    def build(self, var_list):
        """Initialize optimizer variables.


        Args:
          var_list: list of model variables to build Lans variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"))
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"))
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"))

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
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)
        index = self._index_dict[self._var_key(variable)]
        m = self._momentums[index]
        v = self._velocities[index]
        alpha = tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(tf.IndexedSlices(gradient.values * (1 - self.beta_1), gradient.indices))
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(tf.IndexedSlices(tf.square(gradient.values) * (1 - self.beta_2), gradient.indices))
        else:
            # Dense gradients.
            m.assign_add((gradient - m) * (1 - self.beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))
        if self.amsgrad:
            v_hat = self._velocity_hats[index]
            v_hat.assign(tf.maximum(v_hat, v))
            v = v_hat
        denorm = tf.sqrt(v) + self.epsilon
        var_t = (m * alpha) / denorm
        # Apply step weight decay
        use_weight_decay = self._use_weight_decay(variable)
        wd = 0.0
        if use_weight_decay:
            wd = tf.cast(self.weight_decay, variable.dtype)
            var_t = var_t + variable * wd
        if var_t.shape.ndims != 0:
            w_norm = lr * tf.norm(variable, ord=2)
            valid_w_norm = tf.greater(w_norm, 0)
            g_norm = tf.norm(var_t, ord=2)
            ratio_m = tf.where(valid_w_norm, tf.where(tf.greater(g_norm, 0), w_norm / g_norm, lr), lr)
            new_var_t = variable - (ratio_m * self.beta_1 * var_t)
            if isinstance(gradient, tf.IndexedSlices):
                var_t = gradient.values / denorm
            else:
                var_t = gradient / denorm
            if use_weight_decay:
                var_t = var_t + new_var_t * wd
            g_norm = tf.norm(var_t, ord=2)
            ratio_g = tf.where(valid_w_norm, tf.where(tf.greater(g_norm, 0), w_norm / g_norm, lr), lr)
            new_var_t = new_var_t - (ratio_g * (1 - self.beta_1) * var_t)
            variable.assign(new_var_t)
        else:
            variable.assign_sub(lr * var_t)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "weight_decay": self.weight_decay,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
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


Lans.__doc__ = Lans.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
