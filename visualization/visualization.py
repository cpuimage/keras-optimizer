from benchmark_functions import *
import numpy as np

from keras.optimizers.optimizer_experimental.adamw import AdamW
from keras.optimizers.optimizer_experimental.ftrl import Ftrl
from keras.optimizers.optimizer_experimental.adadelta import Adadelta
from keras.optimizers.optimizer_experimental.adam import Adam
from keras.optimizers.optimizer_experimental.adamax import Adamax
from keras.optimizers.optimizer_experimental.nadam import Nadam
from keras.optimizers.optimizer_experimental.rmsprop import RMSprop
from keras.optimizers.optimizer_experimental.sgd import SGD
from keras.optimizers.optimizer_experimental.adagrad import Adagrad
from optimizer.AdaBelief import AdaBelief
from optimizer.AdaFactor import AdaFactor
from optimizer.AdaMomentum import AdaMomentum
from optimizer.Apollo import Apollo
from optimizer.Fromage import Fromage
from optimizer.LAMB import LAMB
from optimizer.Lans import Lans
from optimizer.MadGrad import MadGrad
from optimizer.RectifiedAdam import RectifiedAdam
from optimizer.Rm3 import Rm3
from optimizer.DiffGrad import DiffGrad
from optimizer.nero import nero
from optimizer.VAdam import VAdam
from landscape_3d import landscape_3d

benchmark_fn = [beale(), rosenbrock(), sphere()]
benchmark_index = np.random.randint(len(benchmark_fn))
learning_rate = 0.01
landscape_3d(benchmark_fn[benchmark_index], [
    Adam(learning_rate=learning_rate),
    AdamW(learning_rate=learning_rate),
    Adamax(learning_rate=learning_rate),
    Adagrad(learning_rate=learning_rate),
    Adadelta(learning_rate=learning_rate),
    RMSprop(learning_rate=learning_rate),
    SGD(learning_rate=learning_rate),
    Ftrl(learning_rate=learning_rate),
    Nadam(learning_rate=learning_rate),
    AdaBelief(learning_rate=learning_rate),
    AdaFactor(learning_rate=learning_rate),
    AdaMomentum(learning_rate=learning_rate),
    Apollo(learning_rate=learning_rate),
    Fromage(learning_rate=learning_rate),
    LAMB(learning_rate=learning_rate),
    Lans(learning_rate=learning_rate),
    MadGrad(learning_rate=learning_rate),
    RectifiedAdam(learning_rate=learning_rate),
    Rm3(learning_rate=learning_rate),
    DiffGrad(learning_rate=learning_rate),
    nero(learning_rate=learning_rate),
    VAdam(learning_rate=learning_rate),
])
