from abc import ABC, abstractmethod
import torch.nn as nn


class ModelABC(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.has_loss = False
        self.has_eval = False

    def setup(self, summary_writer, log_freq, **kwargs):
        self.summary = summary_writer
        self.log_freq = log_freq
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _build_loss(self):
        pass

    def _build_evaluation(self):
        pass

    @abstractmethod
    def training_step(self):
        pass

    @abstractmethod
    def on_train_finished(self):
        pass

    @abstractmethod
    def validation_step(self):
        pass

    @abstractmethod
    def on_val_finished(self):
        pass

    @abstractmethod
    def compute_loss(self):
        pass

    @abstractmethod
    def testing_step(self, batch, batch_idx):
        pass

    def inference_step(self, batch, batch_idx):
        pass
