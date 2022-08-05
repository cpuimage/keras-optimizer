"""AdaFactor optimizer implementation."""

import re

import tensorflow as tf
import numpy as np
from keras.optimizers.optimizer_experimental import optimizer


class AdaFactor(optimizer.Optimizer):
    r"""Optimizer that implements the AdaFactor algorithm.

    Adafactor: Adaptive Learning Rates with Sublinear Memory Cost (https://arxiv.org/abs/1804.04235)
    https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/optimize.py
    https://kexue.fm/archives/7302
    """

    def __init__(
            self,
            learning_rate=0.001,
            weight_decay=0.004,
            beta_1=0.0,
            beta_2=0.999,
            epsilon=1e-16,
            min_dim_size_to_factor=128,
            clipping_threshold=1.0,
            multiply_by_parameter_scale=True,
            epsilon_scale=1e-3,
            amsgrad=False,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="AdaFactor",
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
        self.min_dim_size_to_factor = min_dim_size_to_factor
        self.clipping_threshold = clipping_threshold
        self.multiply_by_parameter_scale = multiply_by_parameter_scale
        self.epsilon_scale = epsilon_scale
        self.has_beta_1 = beta_1 > 0.0
        if self.weight_decay is None:
            raise ValueError(
                "Missing value of `weight_decay` which is required and"
                " must be a float value."
            )

    def factored_shape(self, tensor):
        shape = tf.shape(tensor)
        if len(shape) < 2:
            return None
        shape = np.array(shape)
        indices = shape.argpartition(-2)
        if shape[indices[-2]] < self.min_dim_size_to_factor:
            return None
        shape1, shape2 = np.array(shape), np.array(shape)
        shape1[indices[-1]] = 1
        shape2[indices[-2]] = 1
        return shape1, indices[-1], shape2, indices[-2]

    def build(self, var_list):
        """Initialize optimizer variables.

        Args:
          var_list: list of model variables to build AdaFactor variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._velocities = []
        self._velocities_r = []
        self._velocities_c = []
        for var in var_list:
            factored_shape = self.factored_shape(var)
            if factored_shape is None:
                self._velocities.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="v"))
            else:
                shape1, axis1, shape2, axis2 = factored_shape
                self._velocities_r.append(
                    self.add_variable_from_reference(model_variable=var, variable_name='vr', shape=shape1))
                self._velocities_c.append(
                    self.add_variable_from_reference(model_variable=var, variable_name='vc', shape=shape2))

        if self.has_beta_1:
            self._momentums = []
            for var in var_list:
                self._momentums.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="m"))
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"))

    def _use_weight_decay(self, variable):
        exclude_from_weight_decay = getattr(self, "_exclude_from_weight_decay", [])
        exclude_from_weight_decay_names = getattr(
            self, "_exclude_from_weight_decay_names", [])
        if variable in exclude_from_weight_decay:
            return False
        for name in exclude_from_weight_decay_names:
            if re.search(name, variable.name) is not None:
                return False
        return True

    def reduce_rms(self, x):
        return tf.sqrt(tf.reduce_mean(tf.square(x)))

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        clipping_threshold = tf.cast(self.clipping_threshold, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)
        index = self._index_dict[self._var_key(variable)]
        factored_shape = self.factored_shape(variable)
        alpha = tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        # Apply step weight decay
        if self._use_weight_decay(variable):
            wd = tf.cast(self.weight_decay, variable.dtype)
            variable.assign_sub(variable * wd * lr)

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            if self.has_beta_1:
                m = self._momentums[index]
                m.assign_add(-m * (1 - self.beta_1))
                m.scatter_add(tf.IndexedSlices(gradient.values * (1 - self.beta_1), gradient.indices))
            else:
                m = gradient.values
            if factored_shape is None:
                v = self._velocities[index]
                v.assign_add(-v * (1 - self.beta_2))
                v.scatter_add(tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2) + self.epsilon, gradient.indices))
            else:
                shape1, axis1, shape2, axis2 = factored_shape
                g2 = tf.square(gradient.values)
                vr = self._velocities_r[index]
                vc = self._velocities_c[index]
                g2r = tf.reduce_mean(g2, axis=axis1, keepdims=True)
                g2c = tf.reduce_mean(g2, axis=axis2, keepdims=True)
                vr.scatter_update(tf.IndexedSlices(self.beta_2 * vr + (1 - self.beta_2) * g2r, gradient.indices))
                vc.scatter_update(tf.IndexedSlices(self.beta_2 * vc + (1 - self.beta_2) * g2c, gradient.indices))
                v = vr * vc / tf.reduce_mean(vr, axis=axis2, keepdims=True)
        else:
            if self.has_beta_1:
                # Dense gradients.
                m = self._momentums[index]
                m.assign_add((gradient - m) * (1 - self.beta_1))
            else:
                m = gradient
            if factored_shape is None:
                v = self._velocities[index]
                v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2) + self.epsilon)
            else:
                shape1, axis1, shape2, axis2 = factored_shape
                vr = self._velocities_r[index]
                vc = self._velocities_c[index]
                g2 = tf.square(gradient)
                g2r = tf.reduce_mean(g2, axis=axis1, keepdims=True)
                g2c = tf.reduce_mean(g2, axis=axis2, keepdims=True)
                vr.assign(vr, self.beta_2 * vr + (1 - self.beta_2) * g2r)
                vc.assign(vc, self.beta_2 * vc + (1 - self.beta_2) * g2c)
                v = vr * vc / tf.reduce_mean(vr, axis=axis2, keepdims=True)
        if self.amsgrad:
            v_hat = self._velocity_hats[index]
            v_hat.assign(tf.maximum(v_hat, v))
            v = v_hat
        var_t = m * alpha * tf.math.rsqrt(v)
        if self.clipping_threshold is not None:
            var_t = var_t / tf.maximum(clipping_threshold, self.reduce_rms(var_t))
        if self.multiply_by_parameter_scale:
            var_t = var_t * tf.maximum(self.reduce_rms(variable), self.epsilon_scale)
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
                "epsilon_scale": self.epsilon_scale,
                "multiply_by_parameter_scale": self.multiply_by_parameter_scale,
                "min_dim_size_to_factor": self.min_dim_size_to_factor,
                "clipping_threshold": self.clipping_threshold,
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


AdaFactor.__doc__ = AdaFactor.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
