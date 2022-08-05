"""Rm3 optimizer implementation."""

import re

import tensorflow as tf

from keras.optimizers.optimizer_experimental import optimizer


class Rm3(optimizer.Optimizer):
    r"""Optimizer that implements the Rm3 algorithm.
        Weak and Strong Gradient Directions: Explaining Memorization, Generalization, and Hardness of Examples at Scale
        https://arxiv.org/abs/2003.07422v2
        https://github.com/google-research/google-research/tree/master/coherent_gradients/weak_and_strong
     """

    def __init__(
            self,
            learning_rate=0.001,
            weight_decay=2e-4,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="Rm3",
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

        if self.weight_decay is None:
            raise ValueError(
                "Missing value of `weight_decay` which is required and"
                " must be a float value."
            )

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._grads_0 = []
        self._grads_1 = []
        self._grads_2 = []
        for var in var_list:
            self._grads_0.append(self.add_variable_from_reference(model_variable=var, variable_name="g0"))
            self._grads_1.append(self.add_variable_from_reference(model_variable=var, variable_name="g1"))
            self._grads_2.append(self.add_variable_from_reference(model_variable=var, variable_name="g2"))

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
        index = self._index_dict[self._var_key(variable)]
        grads = [self._grads_0, self._grads_1, self._grads_2]
        ring_size = 3
        ring_buffer = grads[self.iterations % ring_size][index]
        if isinstance(gradient, tf.IndexedSlices):
            ring_buffer.scatter_update(tf.IndexedSlices(gradient.values, gradient.indices))
        else:
            ring_buffer.assign(gradient)

        def update_fn():
            gradients = tf.concat(
                [tf.expand_dims(grads[i][index], -1) for i in range(ring_size)], -1)
            sum_gradients = tf.reduce_sum(gradients, -1)
            min_gradients = tf.reduce_min(gradients, -1)
            max_gradients = tf.reduce_max(gradients, -1)
            median_of_3 = sum_gradients - min_gradients - max_gradients
            # Apply step weight decay
            if self._use_weight_decay(variable):
                wd = tf.cast(self.weight_decay, variable.dtype)
                median_of_3 = median_of_3 + variable * wd
            variable.assign_sub(median_of_3 * lr)

        tf.cond(self.iterations >= ring_size, update_fn, lambda: {})

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "weight_decay": self.weight_decay,
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


Rm3.__doc__ = Rm3.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
