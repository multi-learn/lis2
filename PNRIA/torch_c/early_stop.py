import abc
from PNRIA.configs.config import TypedCustomizable

class EarlyStopping(TypedCustomizable, abc.ABC):
    config_schema = {
        'patience': {'type': int, 'default': 5},
        'min_delta': {'type': float, 'default': 0.0}
    }

    @abc.abstractmethod
    def step(self, loss):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

class LossEarlyStopping(EarlyStopping):
    def __init__(self):
        self.counter = 0
        self.best_loss = None

    def step(self, loss):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def reset(self):
        self.counter = 0
        self.best_loss = None
