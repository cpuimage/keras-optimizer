"""RectifiedAdam optimizer implementation."""

import re

import tensorflow as tf

from keras.optimizers.optimizer_experimental import optimizer


class RectifiedAdam(optimizer.Optimizer):
    r"""Variant of the Adam optimizer whose adaptive learning rate is rectified
    so as to have a consistent variance.
    It implements the Rectified Adam (a.k.a. RAdam) proposed by
    Liyuan Liu et al. in [On The Variance Of The Adaptive Learning Rate And Beyond](https://arxiv.org/abs/1908.03265).
    Example of usage:
    ```python
    opt = RectifiedAdam(lr=1e-3)
    ```
    Note: `amsgrad` is not described in the original paper. Use it with
          caution.
    RAdam is not a placement of the heuristic warmup, the settings should be
    kept if warmup has already been employed and tuned in the baseline method.
    You can enable warmup by setting `total_steps` and `warmup_proportion`:
    ```python
    opt = RectifiedAdam(
        lr=1e-3,
        total_steps=10000,
        warmup_proportion=0.1,
        min_lr=1e-5,
    )
    """

    def __init__(
            self,
            learning_rate=0.001,
            weight_decay=0.004,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            sma_threshold=5.0,
            total_steps=0,
            warmup_proportion=0.1,
            min_lr=0.0,
            amsgrad=False,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="RectifiedAdam",
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
        self.sma_threshold = sma_threshold
        self.total_steps = total_steps
        self.warmup_proportion = warmup_proportion
        self.min_lr = min_lr
        if self.weight_decay is None:
            raise ValueError(
                "Missing value of `weight_decay` which is required and"
                " must be a float value."
            )

    def build(self, var_list):
        """Initialize optimizer variables.

        Args:
          var_list: list of model variables to build RectifiedAdam variables on.
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
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
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
        beta_1_t = tf.cast(self.beta_1, variable.dtype)
        beta_2_t = tf.cast(self.beta_2, variable.dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        one_minus_beta_2_t = 1.0 - beta_2_t
        sma_inf = 2.0 / one_minus_beta_2_t - 1.0
        sma_t = (sma_inf - 2.0 * local_step * beta_2_power) / (1.0 - beta_2_power)
        r_t = tf.sqrt((sma_t - 4.0) / (sma_inf - 4.0) * (sma_t - 2.0) / (sma_inf - 2.0) * sma_inf / sma_t)
        sma_threshold = tf.cast(self.sma_threshold, variable.dtype)
        sma_t_ge_sma_threshold = sma_t >= sma_threshold
        if self.total_steps > 0:
            total_steps = tf.cast(self.total_steps, variable.dtype)
            warmup_steps = total_steps * tf.cast(self.warmup_proportion, variable.dtype)
            min_lr = tf.cast(self.min_lr, variable.dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr) / decay_steps
            lr = tf.where(
                local_step <= warmup_steps,
                lr * (local_step / warmup_steps),
                lr + decay_rate * tf.minimum(local_step - warmup_steps, decay_steps),
            )
        index = self._index_dict[self._var_key(variable)]
        m = self._momentums[index]
        v = self._velocities[index]

        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        # Apply step weight decay
        if self._use_weight_decay(variable):
            wd = tf.cast(self.weight_decay, variable.dtype)
            variable.assign_sub(variable * wd * lr)

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                )
            )
        else:
            # Dense gradients.
            m.assign_add((gradient - m) * (1 - self.beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))

        if self.amsgrad:
            v_hat = self._velocity_hats[index]
            v_hat.assign(tf.maximum(v_hat, v))
            v = v_hat
        var_t = tf.where(sma_t_ge_sma_threshold, r_t * m / (tf.sqrt(v) + self.epsilon), m) * alpha
        variable.assign_sub(var_t)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "weight_decay": self.weight_decay,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                "sma_threshold": self.sma_threshold,
                "total_steps": self.total_steps,
                "warmup_proportion": self.warmup_proportion,
                "min_lr": self.min_lr,
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


RectifiedAdam.__doc__ = RectifiedAdam.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
