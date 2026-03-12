# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import OmegaConf, DictConfig
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import math
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchmetrics import MetricCollection

# from emg2qwerty.transformer_encoder import TransformerEncoderModule
from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TransformerEncoder,
    TDSConvEncoder
)
from emg2qwerty.transforms import Transform



# add near imports in lightning.py
from torch import nn

class IdentityWithLengths(nn.Module):
    """Identity module that accepts the same call signature as the Transformer.
    This avoids changing call-sites while doing ablation tests.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor | None = None) -> torch.Tensor:
        # ignore input_lengths but keep the same API
        return x



class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
        train_fraction: float = 1,
        train_seed: int = 26, 
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform


        self.train_fraction = float(train_fraction)
        assert 0.0 < self.train_fraction <= 1.0, "train_fraction must be in (0.0, 1.0]"
        self.train_seed = int(train_seed)


    def setup(self, stage: str | None = None) -> None:
        # build full train ConcatDataset (like you already do)
        full_train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )

        # If fraction == 1.0, keep full dataset; otherwise sample windows across the concatenated dataset.
        if self.train_fraction >= 1.0 or math.isclose(self.train_fraction, 1.0):
            self.train_dataset = full_train_dataset
        else:
            # reproducible random sampling of indices
            total = len(full_train_dataset)
            k = max(1, int(total * self.train_fraction))
            rng = torch.Generator()
            rng.manual_seed(self.train_seed)
            perm = torch.randperm(total, generator=rng)
            selected_idx = perm[:k].tolist()
            self.train_dataset = Subset(full_train_dataset, selected_idx)

        # val and test unchanged (you can keep your current behavior)
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        # transformer: DictConfig,

        use_tds: bool = True, 


    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]


        # # ---------- TDS conv encoder (inserted between frontend and transformer) ----------
        # # instantiate TDS conv encoder using provided block_channels and kernel_width args
        # # block_channels and kernel_width are passed into __init__ already
        # self.tds_encoder = TDSConvEncoder(
        #     num_features=num_features,
        #     block_channels=block_channels,
        #     kernel_width=kernel_width,
        # )

        # # compute how many conv-blocks are applied so we can compute temporal reduction
        # # Each TDSConv2dBlock uses a Conv over time with kernel_size=kernel_width and no padding,
        # # which reduces length by (kernel_width - 1) per block. TDSConvEncoder stacks len(block_channels) blocks.
        # self.tds_num_blocks = len(block_channels)
        # self.tds_time_reduction = self.tds_num_blocks * (kernel_width - 1)
        # # ---------------------------------------------------------------------------------









        self.use_tds = use_tds
        if self.use_tds:
            self.tds_encoder = TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            )
            # compute how many conv-blocks are applied so we can compute temporal reduction
            # Each TDSConv2dBlock uses a Conv over time with kernel_size=kernel_width and no padding,
            # which reduces length by (kernel_width - 1) per block. TDSConvEncoder stacks len(block_channels) blocks.
            self.tds_num_blocks = len(block_channels)
            self.tds_time_reduction = self.tds_num_blocks * (kernel_width - 1)
        else:
            # disabled: use a no-op identity module so call sites don't change
            self.tds_encoder = IdentityWithLengths()
            # no temporal shrinkage when TDS is disabled
            self.tds_num_blocks = 0
            self.tds_time_reduction = 0




























        # ---------------- downsampler config (hardcoded) ----------------
        # Hardcode the downsampling behaviour here (no Hydra / YAML)
        self.downsample = True                 # set False to disable downsampling entirely
        self.downsample_kernel = 3              # kernel size for Conv1d (set 1 for no receptive-field change)
        self.downsample_stride = 2              # stride >1 downsamples, stride=1 preserves temporal length
        self.downsample_padding = 1             # padding for Conv1d

        if self.downsample:
            # Conv1d uses shape (N, channels, T) so channels=num_features
            self.time_downsampler = nn.Conv1d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=self.downsample_kernel,
                stride=self.downsample_stride,
                padding=self.downsample_padding,
            )
        else:
            # keep attribute for code simplicity, but set to None
            self.time_downsampler = None
        # ----------------------------------------------------------------





        # ---------- frontend that prepares (T, N, num_features) ----------
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),  # (T, N, bands=2, C=16, freq)
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),  # -> (T, N, num_bands, mlp_features[-1])
            nn.Flatten(start_dim=2),  # -> (T, N, num_features)
        )

        # ---------- Transformer encoder ----------
        # self.encoder = TransformerEncoder(
        #     d_model=num_features,      # keep embedding dim equal to num_features
        #     nhead=8,                  # tune (must divide d_model)
        #     num_layers=3,             # tune: 2-6 recommended
        #     dim_feedforward=2048,     # tune
        #     dropout=0.1,
        #     max_len=20000,
        # )

        self.encoder = TransformerEncoder(
            d_model=num_features,      # keep embedding dim equal to num_features
            nhead=4,                  # tune (must divide d_model)
            num_layers=2,             # tune: 2-6 recommended
            dim_feedforward=1024,     # tune
            dropout=0.1,
            max_len=20000,
        )
        # temporary ablation: bypass transformer

        # self.encoder = IdentityWithLengths()

        # ---------- head that maps (T, N, num_features) -> (T, N, num_classes) ----------
        self.head = nn.Sequential(
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )




        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        # frontend -> (T, N, num_features)
        x = self.frontend(inputs)  # (T, N, E) where E == num_features

        # -------- apply TDS conv encoder (may reduce temporal length) ----------
        # TDSConvEncoder expects (T, N, num_features) and returns (T_tds, N, num_features)
        x = self.tds_encoder(x)
        # ----------------------------------------------------------------------

        emission_lengths = None
        if input_lengths is not None:
            # First, account for TDS conv temporal shrinkage:
            tds_reduction = getattr(self, "tds_time_reduction", 0)
            tds_out_lengths = (input_lengths - tds_reduction).clamp(min=1)

            if self.downsample and self.time_downsampler is not None:
                k = self.downsample_kernel
                p = self.downsample_padding
                s = self.downsample_stride
                emission_lengths = ((tds_out_lengths + 2 * p - (k - 1) - 1) // s) + 1
                emission_lengths = emission_lengths.clamp(min=1).to(device=input_lengths.device).long()
            else:
                emission_lengths = tds_out_lengths.to(device=inputs.device).long()

        # optionally downsample in time (Conv1d) — same as before
        if self.downsample and self.time_downsampler is not None:
            # (T_tds, N, E) -> (N, E, T_tds) -> Conv1d -> (N, E, T_out) -> (T_out, N, E)
            x = x.permute(1, 2, 0)                 # (N, E, T)
            x = self.time_downsampler(x)           # (N, E, T_out)
            x = x.permute(2, 0, 1)                 # (T_out, N, E)
        else:
            # keep as (T_tds, N, E)
            pass

        # Pass through transformer encoder: pass emission_lengths (may equal input_lengths if not downsampling)
        x = self.encoder(x, input_lengths=emission_lengths)  # (T_out, N, E) or (T_tds, N, E)

        # Classification head -> (T_out_or_T_tds, N, num_classes)
        emissions = self.head(x)  # (T_out, N, num_classes)
        # after emissions = self.head(x)
        if emission_lengths is not None:
            assert emissions.shape[0] >= int(emission_lengths.max()), "emissions shorter than emission_lengths"
        return emissions, emission_lengths


    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size


        

        # --- NEW: Generate the initial padding mask ---
        # T is the first dim, N is the second
        # T_max, N = inputs.shape[0], inputs.shape[1]
        # device = inputs.device
        # # Create mask: True for padding (where index >= length)
        # ids = torch.arange(T_max, device=device).unsqueeze(0) # (1, T)
        # src_key_padding_mask = ids >= input_lengths.unsqueeze(1) # (N, T)

        
        # inside _step()
        emissions, emission_lengths = self.forward(inputs, input_lengths=input_lengths)

        # Ensure emission_lengths exists and on same device
        if emission_lengths is None:
            emission_lengths = torch.full((N,), emissions.shape[0], dtype=torch.long, device=inputs.device)
        else:
            emission_lengths = emission_lengths.to(inputs.device)

        # Sanity check
        assert emissions.shape[0] >= int(emission_lengths.max()), (
            f"Emissions time dim {emissions.shape[0]} < max emission length {int(emission_lengths.max())}"
        )

        loss = self.ctc_loss(
            log_probs=emissions,                         # (T_out, N, num_classes)
            targets=targets.transpose(0, 1),             # (N, T_targets)
            input_lengths=emission_lengths,              # (N,)
            target_lengths=target_lengths,               # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
