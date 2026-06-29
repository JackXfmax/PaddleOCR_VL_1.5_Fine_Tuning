import os
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from nltk import edit_distance

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.optim import create_optimizer_v2

from strhub.data.utils import BaseTokenizer, CharsetAdapter, CTCTokenizer, Tokenizer

def _load_charset(charset: str) -> str:
    if os.path.isfile(charset):
        with open(charset, "r", encoding="utf-8") as f:
            chars = [line.rstrip("\n") for line in f]
        return "".join(chars)
    return charset



@dataclass
class BatchResult:
    num_samples: int
    correct: int
    ned: float
    confidence: float
    label_length: int
    loss: Tensor
    loss_numel: int


EPOCH_OUTPUT = list[dict[str, BatchResult]]


class BaseSystem(pl.LightningModule, ABC):

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        charset_test: str,
        batch_size: int,
        lr: float,
        warmup_pct: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.charset_adapter = CharsetAdapter(charset_test)
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_pct = warmup_pct
        self.weight_decay = weight_decay
        self.outputs: EPOCH_OUTPUT = []

    @abstractmethod
    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        """Inference

        Args:
            images: Batch of images. Shape: N, Ch, H, W
            max_length: Max sequence length of the output. If None, will use default.

        Returns:
            logits: N, L, C (L = sequence length, C = number of classes, typically len(charset_train) + num specials)
        """
        pass

    def _get_optimizer(self) -> Optimizer:
        return create_optimizer_v2(self, 'adamw', self.lr, self.weight_decay)

    def _get_lr_scheduler(self, optimizer: Optimizer) -> OneCycleLR:
        return OneCycleLR(optimizer, self.lr, self.trainer.estimated_stepping_batches,
                          pct_start=self.warmup_pct, anneal_strategy='linear',
                          div_factor=25, final_div_factor=1e4)

    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        scheduler = self._get_lr_scheduler(optimizer)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def _eval_step(self, batch, validation: bool) -> Optional[BatchResult]:
        images, labels = batch
        correct = 0
        total = 0
        ned = 0
        confidence = 0
        label_length = 0
        if validation:
            images = images.to(self.device)
        logits, loss, loss_numel = self.forward_logits_loss(images, labels)
        probs = logits.softmax(-1)
        preds, confidences = self.tokenizer.decode(probs)
        for pred, label in zip(preds, labels):
            confidence += confidences[pred].mean().item()
            pred = self.charset_adapter(pred)
            label_length += len(label)
            ned += edit_distance(pred, label) / max(len(pred), len(label))
            if pred == label:
                correct += 1
            total += 1
        return BatchResult(total, correct, ned, confidence, label_length, loss, loss_numel)

    def _aggregate_results(self, outputs: EPOCH_OUTPUT) -> tuple[float, float, float]:
        if not outputs:
            return 0.0, 0.0, 0.0
        counts = torch.as_tensor([o['batch'].num_samples for o in outputs])
        corrects = torch.as_tensor([o['batch'].correct for o in outputs])
        neds = torch.as_tensor([o['batch'].ned for o in outputs])
        losses = torch.as_tensor([o['batch'].loss for o in outputs])
        loss_numels = torch.as_tensor([o['batch'].loss_numel for o in outputs])
        acc = corrects.sum() / counts.sum()
        ned = neds.sum() / counts.sum()
        loss = losses.sum() / loss_numels.sum()
        return acc, ned, loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self._eval_step(batch, True)

    def on_validation_epoch_end(self) -> None:
        acc, ned, loss = self._aggregate_results(self.outputs)
        self.outputs.clear()
        self.log('val_accuracy', 100 * acc, sync_dist=True)
        self.log('val_NED', 100 * ned, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)
        self.log('hp_metric', acc, sync_dist=True)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self._eval_step(batch, False)


class CrossEntropySystem(BaseSystem):

    def __init__(
        self, charset_train: str, charset_test: str, batch_size: int, lr: float, warmup_pct: float, weight_decay: float
    ) -> None:
        charset_train = _load_charset(charset_train)
        charset_test = _load_charset(charset_test)
        tokenizer = Tokenizer(charset_train)
        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id

    def forward_logits_loss(self, images: Tensor, labels: list[str]) -> tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.encode(labels, self.device)
        targets = targets[:, 1:]  # Discard <bos>
        max_len = targets.shape[1] - 1  # exclude <eos> from count
        logits = self.forward(images, max_len)
        loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        loss_numel = (targets != self.pad_id).sum()
        return logits, loss, loss_numel


class CTCSystem(BaseSystem):

    def __init__(
        self, charset_train: str, charset_test: str, batch_size: int, lr: float, warmup_pct: float, weight_decay: float
    ) -> None:
        charset_train = _load_charset(charset_train)
        charset_test = _load_charset(charset_test)
        tokenizer = CTCTokenizer(charset_train)
        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.blank_id = tokenizer.blank_id

    def forward_logits_loss(self, images: Tensor, labels: list[str]) -> tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.encode(labels, self.device)
        logits = self.forward(images)
        log_probs = logits.log_softmax(-1).transpose(0, 1)  # swap batch and seq. dims
        T, N, _ = log_probs.shape
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=self.device)
        target_lengths = torch.as_tensor(list(map(len, labels)), dtype=torch.long, device=self.device)
        loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=self.blank_id, zero_infinity=True)
        return logits, loss, N
