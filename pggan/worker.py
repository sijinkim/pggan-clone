import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pggan.types import (
    NeuralNet,
    Optimizer,
    StateDict,
)


class Worker:
    def __init__(
        self,
        dataset: Dataset
    ) -> None:
        self.dataset = dataset
        self.dataloader = self._build_data_loader()

    def _build_data_loader(self):
        return DataLoader(
            self.dataset,
        )


class Trainer(Worker):
    def __init__(
        self,
        dataset:        Dataset,
        generator:      NeuralNet,
        discriminator:  NeuralNet,
        optimizer:      Optimizer,
    ) -> None:
        super(Trainer, self).__init__(dataset)
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer = optimizer

    def train(self):
        for x in self.dataloader:
            yield x.sum()

    def get_state(self) -> StateDict:
        return {
            'model': {
                'generator': self.generator.state_dict(),
                'discriminator': self.discriminator.state_dict(),
            },
            'optimizer': self.optimizer.state_dict(),
        }


class Validator(Worker):
    def __init__(
        self,
        dataset: Dataset,
    ) -> None:
        super(Validator, self).__init__(dataset)
        self.best_loss = np.Inf

    def validate(self):
        for x in self.dataloader:
            yield x.sum()

    def is_best_score(self, loss):
        if self.best_loss > loss:
            self.best_loss = loss
            return True
        return False
