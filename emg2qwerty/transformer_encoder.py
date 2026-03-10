# emg2qwerty/transformer_encoder.py
import math
from typing import Optional, Dict

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, E)
        T = x.size(0)
        return x + self.pe[:T].unsqueeze(1)


class ConvSubsampler(nn.Module):
    """Conv1D subsampler: reduces time length and projects channels."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        # Conv1d expects (N, in_dim, T)
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=0)
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, E)
        x = x.permute(1, 2, 0)  # -> (N, E, T)
        x = self.conv(x)        # -> (N, out_dim, T_out)
        x = x.permute(2, 0, 1)  # -> (T_out, N, out_dim)
        return x

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        # input_lengths: (N,) ints
        # formula: floor((L - (kernel_size - 1) - 1) / stride) + 1  => floor((L - kernel + 1) / stride)
        # More simply: (L - kernel_size) // stride + 1, but handle small L
        L = input_lengths.clone()
        L = torch.where(L < self.kernel_size, torch.zeros_like(L), L)
        return torch.div(L - self.kernel_size, self.stride, rounding_mode="floor") + 1


class TransformerEncoderModule(nn.Module):
    """
    Transformer encoder module with optional Conv subsampler.

    Input: (T, N, E)  -> Output: (T_out, N, E)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int ,
        num_layers: int ,
        dim_feedforward: int ,
        dropout: float ,
        max_len: int ,
        use_pos_enc: bool ,
        subsample: Optional[Dict] ,  # {"out_dim": d_model, "kernel":3, "stride":2}
        layer_norm_eps: float ,
    ):
        super().__init__()



        # --- ADD THIS LINE TO INITIALIZE THE ATTRIBUTE ---
        self._proj_in = None 
        
        self.subsampler = None
        if subsample is not None:
            out_dim = subsample.get("out_dim", d_model)
            kernel = int(subsample.get("kernel", 3))
            stride = int(subsample.get("stride", 2))
            self.subsampler = ConvSubsampler(in_dim=d_model, out_dim=out_dim, kernel_size=kernel, stride=stride)
            
            self._proj_back = None
            if out_dim != d_model:
                self._proj_back = nn.Linear(out_dim, d_model)
        else:
            self._proj_back = None
            # Optional: If your MLP output doesn't match d_model, 
            # you could initialize self._proj_in = nn.Linear(mlp_dim, d_model) here.

        self.use_pos_enc = use_pos_enc




        self.subsampler = None
        if subsample is not None:
            out_dim = subsample.get("out_dim", d_model)
            kernel = int(subsample.get("kernel", 3))
            stride = int(subsample.get("stride", 2))
            self.subsampler = ConvSubsampler(in_dim=d_model, out_dim=out_dim, kernel_size=kernel, stride=stride)
            # if output dim differs from d_model, add a projection after transformer to d_model
            self._proj_back = None
            if out_dim != d_model:
                self._proj_back = nn.Linear(out_dim, d_model)
        else:
            self._proj_back = None

        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        
    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (T, N, E)
        src_key_padding_mask: (N, T) boolean mask (True for padding positions)
        returns: (T_out, N, E)
        """
        # If we have a subsampler, apply it first so we know T_out
        if self.subsampler is not None:
            x = self.subsampler(x)  # -> (T_out, N, out_dim) maybe
            if self._proj_back is not None:
                x = self._proj_back(x)  # project back to d_model if needed

            # --- NEW: downsample the padding mask to match T_out ---
            if src_key_padding_mask is not None:
                # src_key_padding_mask expected shape: (N, T_in)
                if src_key_padding_mask.dim() != 2:
                    raise ValueError("src_key_padding_mask must be (N, T) boolean")

                # compute original valid lengths (count of non-padding positions)
                # src_key_padding_mask uses True for padding positions
                valid_lengths = (~src_key_padding_mask).sum(dim=1)  # (N,) ints

                # compute new lengths after subsampling
                new_lengths = self.subsampler.output_lengths(valid_lengths)  # (N,)

                T_out = x.size(0)  # sequence length after subsampler

                # build new padding mask: True where index >= new_length
                device = src_key_padding_mask.device
                idx = torch.arange(T_out, device=device).unsqueeze(0)  # (1, T_out)
                new_mask = idx >= new_lengths.unsqueeze(1)  # (N, T_out), bool

                src_key_padding_mask = new_mask
            # --- END NEW ---
        else:
            # no subsampler: optionally handle case where input feature dim != d_model
            if self._proj_in is not None:
                x = self._proj_in(x)

        if self.use_pos_enc:
            x = self.pos_enc(x)

        # transformer expects (S, N, E)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.layer_norm(x)
        return x

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute output lengths given input_lengths (N,). If no subsampler, returns input_lengths."""
        if self.subsampler is None:
            return input_lengths
        return self.subsampler.output_lengths(input_lengths)