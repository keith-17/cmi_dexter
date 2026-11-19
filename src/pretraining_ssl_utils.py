"""pretraining_ssl_utils.py -- PyTorch pretraining & self-supervised masked-signal SSL for MultiBranchTSAI.

Supports two pretraining regimes prior to downstream target fine-tuning on the
``MultiBranchTSAINet`` multimodal backbone (``src/utils.py``):

1. **Supervised Pretraining (``pretraining_mode='supervised'``)**:
   Trains a temporary classification head + shared ``MultiBranchTSAINet`` encoders/fusion
   on a broader source task (e.g., all 18 fine-grained ``gesture`` labels or baseline
   recordings), transfers every backbone/fusion/branch weight to the downstream
   classifier, and optionally warms up the new downstream classification head
   (``finetuning_head_epochs``) while backbone weights are frozen.

2. **Masked-Signal Self-Supervised Learning (``pretraining_mode='masked_ssl'``)**:
   Masks contiguous temporal spans (``ssl_span_length``, ``ssl_mask_ratio``) directly in
   the raw multimodal sensor recordings BEFORE recomputing derived channels, static
   features, and branch tensors (preventing feature leakage from uncorrupted signal
   statistics, while strictly preserving unit-norm quaternions ``[1, 0, 0, 0]`` on masked
   rotation rows). A lightweight temporal reconstruction decoder
   (``MaskedSSLReconstructionHead``) reconstructs the standardized target signal at the
   hidden, originally valid timesteps via ``nn.MSELoss`` or ``nn.SmoothL1Loss``. The
   trained encoder/fusion weights are then transferred to the downstream classifier.

When ``pretraining_epochs == 0`` and ``finetuning_head_epochs == 0`` (the default in
``MultiBranchTSAIClassifier``), zero pretraining steps run and zero extra computation is
incurred.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn

try:
    from .base_utils import (
        ACC_COLUMNS,
        MODALITIES,
        ROT_COLUMNS,
        SEQ_COL,
        THM_COLUMNS,
        TOF_COLUMNS,
        InvalidParams,
    )
except ImportError:
    from base_utils import (
        ACC_COLUMNS,
        MODALITIES,
        ROT_COLUMNS,
        SEQ_COL,
        THM_COLUMNS,
        TOF_COLUMNS,
        InvalidParams,
    )

PRETRAINING_MODES: Tuple[str, ...] = ("none", "supervised", "masked_ssl")
PRETRAINING_SOURCES: Tuple[str, ...] = ("all", "baseline", "non_target", "target")
SSL_LOSSES: Tuple[str, ...] = ("mse", "huber")

_DEFAULT_UNIT_QUATERNION: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def select_pretraining_sequences(
    X: pd.DataFrame,
    y: Any = None,
    *,
    source: str = "all",
    sequence_col: str = SEQ_COL,
) -> pd.DataFrame:
    """Select the subset of recordings used during Stage-1 pretraining."""
    src = str(source or "all").strip().lower()
    if src not in PRETRAINING_SOURCES:
        raise InvalidParams(f"pretraining_source must be one of {PRETRAINING_SOURCES}, got {source!r}")
    if src == "all" or len(X) == 0:
        return X

    seq_ids = X[sequence_col].astype(str)
    meta_df: Optional[pd.DataFrame] = None
    if isinstance(y, pd.DataFrame) and sequence_col in y.columns:
        meta_df = y.drop_duplicates(sequence_col).set_index(y.drop_duplicates(sequence_col)[sequence_col].astype(str))
    elif any(c in X.columns for c in ("is_target", "bfrb")):
        meta_df = X.drop_duplicates(sequence_col).set_index(X.drop_duplicates(sequence_col)[sequence_col].astype(str))

    if meta_df is None:
        return X

    keep_ids: Optional[pd.Index] = None
    if "is_target" in meta_df.columns:
        flags = pd.to_numeric(meta_df["is_target"], errors="coerce")
        if src in {"baseline", "non_target"}:
            keep_ids = meta_df.index[flags == 0]
        elif src == "target":
            keep_ids = meta_df.index[flags == 1]
    elif "bfrb" in meta_df.columns:
        vals = meta_df["bfrb"].astype(str)
        if src in {"baseline", "non_target"}:
            keep_ids = meta_df.index[vals == "non_bfrb"]
        elif src == "target":
            keep_ids = meta_df.index[vals != "non_bfrb"]

    if keep_ids is None or len(keep_ids) == 0:
        # Safe fallback to all available fit-partition sequences if subset is empty
        return X
    return X.loc[seq_ids.isin(set(keep_ids.astype(str)))].copy()


def transfer_torch_backbone_weights(
    source_model: nn.Module,
    dest_model: nn.Module,
    *,
    exclude_prefixes: Tuple[str, ...] = ("classifier",),
) -> Tuple[List[str], List[str]]:
    """Copy all matching parameters & BatchNorm buffers except head layers from ``source_model`` to ``dest_model``.

    Raises ``ValueError`` if a non-head parameter/buffer is missing or shape-mismatched
    or if no weights could be transferred.
    """
    src_state = source_model.state_dict()
    dst_state = dest_model.state_dict()
    transferred: List[str] = []
    skipped: List[str] = []

    updated = dict(dst_state)
    for name, dst_tensor in dst_state.items():
        if any(name == p or name.startswith(p + ".") for p in exclude_prefixes):
            skipped.append(name)
            continue
        if name not in src_state:
            raise ValueError(f"Pretraining source model is missing backbone key {name!r}.")
        src_tensor = src_state[name]
        if tuple(src_tensor.shape) != tuple(dst_tensor.shape):
            raise ValueError(
                f"Backbone tensor shape mismatch at {name!r}: "
                f"source={tuple(src_tensor.shape)} vs dest={tuple(dst_tensor.shape)}"
            )
        updated[name] = src_tensor.detach().clone().to(device=dst_tensor.device, dtype=dst_tensor.dtype)
        transferred.append(name)

    if not transferred:
        raise ValueError("No backbone weights were transferred during pretraining handoff.")
    dest_model.load_state_dict(updated, strict=True)
    return transferred, skipped


class MaskedSSLReconstructionHead(nn.Module):
    """Per-modality temporal reconstruction decoder for Masked-Signal SSL pretraining.

    Conditions learnable per-timestep queries purely on the shared branch embedding and
    multimodal fused latent representation (no uncorrupted input shortcut) so 100% of
    reconstruction gradients train the ``MultiBranchTSAINet`` branch encoders and fusion
    layers across every ``tsai`` backbone architecture.
    """

    def __init__(
        self,
        branch_specs: Mapping[str, Tuple[int, int]],
        branch_units: int,
        fusion_units: int,
        *,
        decoder_units: int = 64,
        decoder_dropout: float = 0.1,
    ):
        super().__init__()
        self.branch_order = list(branch_specs.keys())
        d_dec = max(16, int(decoder_units) if int(decoder_units) > 0 else 64)
        drop = float(decoder_dropout)
        cond_in = int(branch_units) + int(fusion_units)

        self.pos_embs = nn.ParameterDict()
        self.cond_projs = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for name, (seq_len, n_ch) in branch_specs.items():
            t_len = max(1, int(seq_len))
            self.pos_embs[name] = nn.Parameter(torch.randn(1, t_len, d_dec) * 0.02)
            self.cond_projs[name] = nn.Sequential(
                nn.Linear(cond_in, d_dec),
                nn.GELU(),
            )
            self.decoders[name] = nn.Sequential(
                nn.Conv1d(d_dec, d_dec, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Dropout(drop) if drop > 0 else nn.Identity(),
                nn.Conv1d(d_dec, int(n_ch), kernel_size=1),
            )

    def forward(
        self,
        branch_embs: Mapping[str, torch.Tensor],
        fused: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        recons: Dict[str, torch.Tensor] = {}
        for name in self.branch_order:
            cond = self.cond_projs[name](torch.cat([branch_embs[name], fused], dim=-1))
            h = self.pos_embs[name] + cond.unsqueeze(1)
            rec = self.decoders[name](h.transpose(1, 2)).transpose(1, 2)
            recons[name] = rec
        return recons


def _modality_raw_column_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    imu_cols = [c for c in ACC_COLUMNS + ROT_COLUMNS if c in df.columns]
    thm_cols = [c for c in THM_COLUMNS if c in df.columns]
    tof_cols = [c for c in TOF_COLUMNS if c in df.columns]
    groups: Dict[str, List[str]] = {}
    if imu_cols:
        groups["imu"] = imu_cols
    if thm_cols:
        groups["thm"] = thm_cols
    if tof_cols:
        groups["tof"] = tof_cols
    return groups


def corrupt_raw_dataframe_spans(
    X: pd.DataFrame,
    *,
    sequence_col: str = SEQ_COL,
    mask_ratio: float = 0.25,
    span_length: int = 10,
    mask_modalities_together: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Mask contiguous temporal spans in raw sensor columns prior to feature extraction.

    Quaternion rows ``rot_w/x/y/z`` chosen for masking are replaced with the unit
    identity quaternion ``(1, 0, 0, 0)`` so unit-norm invariants and derived rotation
    physics remain valid, while all other masked sensor values are zeroed out before
    re-running ``extractor.featurize``.
    """
    corrupted = X.copy()
    col_groups = _modality_raw_column_groups(corrupted)
    if not col_groups or len(corrupted) == 0:
        return corrupted

    ratio = float(np.clip(mask_ratio, 0.0, 0.95))
    span = max(1, int(span_length))
    if ratio <= 0.0:
        return corrupted

    rng = np.random.default_rng(int(seed))
    seq_indices = corrupted.groupby(sequence_col, sort=False).indices
    rot_present = [c for c in ROT_COLUMNS if c in corrupted.columns]

    if mask_modalities_together:
        merged_cols = [c for cols in col_groups.values() for c in cols]
        group_iter: Sequence[Tuple[str, List[str]]] = [("all", merged_cols)]
    else:
        group_iter = list(col_groups.items())

    for _, positions in seq_indices.items():
        pos_arr = np.asarray(positions, dtype=int)
        n_rows = len(pos_arr)
        if n_rows < 2:
            continue
        for _, cols in group_iter:
            needed = max(1, int(math.ceil(n_rows * ratio)))
            chosen = np.zeros(n_rows, dtype=bool)
            attempts = 0
            while int(chosen.sum()) < needed and attempts < max(4, n_rows * 3):
                avail = np.flatnonzero(~chosen)
                if len(avail) == 0:
                    break
                start = int(rng.choice(avail))
                chosen[start:min(n_rows, start + span)] = True
                attempts += 1
            if not chosen.any():
                continue
            target_rows = corrupted.index[pos_arr[chosen]]
            non_rot = [c for c in cols if c not in ROT_COLUMNS]
            if non_rot:
                corrupted.loc[target_rows, non_rot] = 0.0
            rot_in_group = [c for c in cols if c in rot_present]
            if rot_in_group:
                for q_i, q_col in enumerate(ROT_COLUMNS):
                    if q_col in rot_in_group:
                        corrupted.loc[target_rows, q_col] = _DEFAULT_UNIT_QUATERNION[q_i]

    return corrupted


def build_ssl_mask_weights(
    corrupted_chunks: Mapping[str, Any],
    clean_chunks: Mapping[str, Any],
    *,
    mask_ratio: float = 0.25,
    span_length: int = 10,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Produce per-branch boolean hidden masks matching ``corrupted_chunks`` vs ``clean_chunks``.

    Ensures non-empty valid hidden positions on every modality even when a sensor column
    happened to sit at ``0.0`` prior to masking, while strictly excluding padded/absent steps.
    """
    rng = np.random.default_rng(int(seed))
    mod_idx = {"imu": 0, "thm": 1, "tof": 2}
    hidden_masks: Dict[str, np.ndarray] = {}
    targets: Dict[str, np.ndarray] = {}

    span = max(1, int(span_length))
    ratio = float(np.clip(mask_ratio, 0.0, 0.95))

    for name, clean_arr in clean_chunks["branches"].items():
        corrupt_arr = corrupted_chunks["branches"][name]
        b_mask = clean_chunks["masks"][name].astype(bool)
        pres = clean_chunks["present"][:, mod_idx[name]].astype(bool)[:, None]
        valid_steps = b_mask & pres  # (N_win, T_mod)

        diff = np.abs(corrupt_arr - clean_arr).max(axis=-1) > 1e-6  # (N_win, T_mod)
        hidden = diff & valid_steps

        # Guarantee valid hidden positions in windows where raw constant signals masked identically
        n_win, t_len = hidden.shape
        if n_win > 0 and ratio > 0 and not hidden.any():
            for w in range(n_win):
                if not valid_steps[w].any():
                    continue
                valid_pos = np.flatnonzero(valid_steps[w])
                s = int(rng.choice(valid_pos))
                hidden[w, s:min(t_len, s + span)] = True
            hidden &= valid_steps

        hidden_masks[name] = hidden[:, :, None].astype(np.float32) * valid_steps[:, :, None].astype(np.float32)
        targets[name] = clean_arr.astype(np.float32)

    return hidden_masks, targets


def _masked_ssl_reconstruction_loss(
    recons: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    weights: Mapping[str, torch.Tensor],
    *,
    loss_name: str = "mse",
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    use_huber = str(loss_name).lower() == "huber"
    for name, pred in recons.items():
        w = weights[name]  # (B, T, 1)
        denom = w.sum() * float(pred.shape[-1])
        if float(denom.detach().cpu().item()) <= 0.0:
            continue
        tgt = targets[name]
        err = nn.functional.smooth_l1_loss(pred, tgt, reduction="none") if use_huber else (pred - tgt) ** 2
        losses.append((err * w).sum() / denom.clamp_min(1.0))
    if not losses:
        first_pred = next(iter(recons.values()))
        return (first_pred * 0.0).sum()
    return torch.stack(losses).mean()


class PretrainingAndSSLMixin:
    """Mixin adding PyTorch Supervised Pretraining & Masked-Signal SSL to ``MultiBranchTSAIClassifier``.

    All pretraining and SSL knobs default to ``0`` / ``0.0`` / ``"none"`` / ``False`` so
    standard downstream training has zero overhead unless explicitly enabled.
    """

    def validate_pretraining_params(self) -> None:
        mode = str(self.pretraining_mode or "none").strip().lower()
        if mode not in PRETRAINING_MODES:
            raise InvalidParams(f"pretraining_mode must be one of {PRETRAINING_MODES}, got {self.pretraining_mode!r}")
        src = str(self.pretraining_source or "all").strip().lower()
        if src not in PRETRAINING_SOURCES:
            raise InvalidParams(f"pretraining_source must be one of {PRETRAINING_SOURCES}, got {self.pretraining_source!r}")
        if int(self.pretraining_epochs) < 0:
            raise InvalidParams("pretraining_epochs must be >= 0")
        if int(self.finetuning_head_epochs) < 0:
            raise InvalidParams("finetuning_head_epochs must be >= 0")
        if float(self.pretraining_learning_rate) < 0:
            raise InvalidParams("pretraining_learning_rate must be >= 0")
        if int(self.pretraining_batch_size) < 0:
            raise InvalidParams("pretraining_batch_size must be >= 0")
        if not 0.0 <= float(self.pretraining_validation_split) < 0.9:
            raise InvalidParams("pretraining_validation_split must be in [0, 0.9)")
        if int(self.pretraining_early_stopping_patience) < 0:
            raise InvalidParams("pretraining_early_stopping_patience must be >= 0")
        if not 0.0 <= float(self.ssl_mask_ratio) < 1.0:
            raise InvalidParams("ssl_mask_ratio must be in [0, 1)")
        if int(self.ssl_span_length) < 0:
            raise InvalidParams("ssl_span_length must be >= 0")
        if int(self.ssl_decoder_units) < 0:
            raise InvalidParams("ssl_decoder_units must be >= 0")
        if not 0.0 <= float(self.ssl_decoder_dropout) < 1.0:
            raise InvalidParams("ssl_decoder_dropout must be in [0, 1)")
        if str(self.ssl_loss_name).lower() not in SSL_LOSSES:
            raise InvalidParams(f"ssl_loss_name must be one of {SSL_LOSSES}, got {self.ssl_loss_name!r}")

    def _effective_pretraining_lr(self) -> float:
        lr = float(self.pretraining_learning_rate)
        return lr if lr > 0.0 else float(self.learning_rate)

    def _effective_pretraining_batch_size(self) -> int:
        bs = int(self.pretraining_batch_size)
        return bs if bs > 0 else max(1, int(self.batch_size))

    def _extract_named_sequence_labels(self, X: pd.DataFrame, y: Any, column: str) -> pd.Series:
        seq_col = self.extractor_.sequence_col
        seq_order = pd.Index(X[seq_col].astype(str).unique(), name=seq_col)
        if isinstance(y, pd.DataFrame) and column in y.columns:
            frame = y.drop_duplicates(seq_col) if seq_col in y.columns else y
            if seq_col in frame.columns:
                aligned = frame.set_index(frame[seq_col].astype(str))[column].reindex(seq_order)
            else:
                aligned = pd.Series(y[column].to_numpy()[:len(seq_order)], index=seq_order)
        elif column in X.columns:
            frame = X.drop_duplicates(seq_col)
            aligned = frame.set_index(frame[seq_col].astype(str))[column].reindex(seq_order)
        else:
            raise InvalidParams(f"Supervised pretraining requires column {column!r} in X or y.")
        if aligned.isna().any():
            raise InvalidParams(f"Supervised pretraining found missing labels in column {column!r}.")
        return aligned.astype(str)

    def _run_pretraining_and_warmup_stages(
        self,
        X_fit: pd.DataFrame,
        y: Any,
        train_chunks: Mapping[str, Any],
        y_train_win: np.ndarray,
        branch_specs: Mapping[str, Tuple[int, int]],
        static_dim: int,
        rng: np.random.Generator,
        criterion: nn.Module,
        val_chunks: Optional[Mapping[str, Any]],
        y_val_win: Optional[np.ndarray],
    ) -> None:
        """Execute Stage-1 pretraining (if enabled) and optional frozen-head warmup."""
        self.pretraining_history_ = {}
        self.finetuning_head_history_ = {}
        self.pretraining_completed_ = False
        self.transferred_layer_names_ = []

        pre_epochs = int(self.pretraining_epochs)
        mode = str(self.pretraining_mode or "none").strip().lower()
        if pre_epochs > 0 and mode != "none":
            if mode == "supervised":
                self._run_supervised_pretrain(X_fit, y, branch_specs, static_dim, rng, pre_epochs)
            elif mode == "masked_ssl":
                self._run_masked_ssl_pretrain(X_fit, y, branch_specs, static_dim, rng, pre_epochs)
            self.pretraining_completed_ = True

        head_epochs = int(self.finetuning_head_epochs)
        if head_epochs > 0:
            self._run_head_warmup(
                train_chunks, y_train_win, rng, criterion, head_epochs,
                val_chunks=val_chunks, y_val_win=y_val_win,
            )

    def _run_supervised_pretrain(
        self,
        X_fit: pd.DataFrame,
        y: Any,
        branch_specs: Mapping[str, Tuple[int, int]],
        static_dim: int,
        rng: np.random.Generator,
        pre_epochs: int,
    ) -> None:
        X_src = select_pretraining_sequences(
            X_fit, y, source=self.pretraining_source, sequence_col=self.extractor_.sequence_col,
        )
        col = str(self.pretraining_label_column or self.primary_target)
        seq_labels = self._extract_named_sequence_labels(X_src, y, col)
        pre_le = LabelEncoder().fit(seq_labels.to_numpy())
        n_pre_classes = max(2, len(pre_le.classes_))

        src_chunks = self.extractor_.transform_chunks(X_src)
        y_src_win = pre_le.transform(
            pd.Series(src_chunks["sequence_ids"].astype(str)).map(seq_labels).to_numpy()
        )

        src_net = self._build_net(branch_specs, static_dim, n_pre_classes)
        opt = torch.optim.AdamW(
            src_net.parameters(), lr=self._effective_pretraining_lr(), weight_decay=float(self.weight_decay),
        )
        loss_fn = nn.CrossEntropyLoss(label_smoothing=float(self.label_smoothing))

        n_samples = len(y_src_win)
        bs = self._effective_pretraining_batch_size()
        hist: Dict[str, List[float]] = {"loss": [], "accuracy": []}

        for epoch in range(1, pre_epochs + 1):
            src_net.train()
            order = rng.permutation(n_samples)
            ep_loss = 0.0
            ep_corr = 0
            ep_tot = 0
            for start in range(0, n_samples, bs):
                idx = order[start:start + bs]
                if len(idx) == 1 and n_samples > 1:
                    idx = np.r_[idx, order[0]]
                b_br, b_mk, b_pr, b_st = self._slice_batch(src_chunks, idx)
                b_y = torch.as_tensor(y_src_win[idx], dtype=torch.long, device=self.device_)
                opt.zero_grad(set_to_none=True)
                logits = src_net(b_br, b_mk, b_pr, b_st)
                loss = loss_fn(logits, b_y)
                loss.backward()
                if self.gradient_clip_norm is not None and float(self.gradient_clip_norm) > 0:
                    nn.utils.clip_grad_norm_(src_net.parameters(), float(self.gradient_clip_norm))
                opt.step()
                ep_loss += float(loss.detach().cpu().item()) * len(idx)
                ep_corr += int((logits.detach().argmax(dim=-1) == b_y).sum().cpu().item())
                ep_tot += len(idx)
            hist["loss"].append(ep_loss / max(1, ep_tot))
            hist["accuracy"].append(ep_corr / max(1, ep_tot))
            if self.verbose and int(self.verbose) > 0:
                print(
                    f"  [supervised-pretrain {epoch}/{pre_epochs}] "
                    f"loss={hist['loss'][-1]:.4f} acc={hist['accuracy'][-1]:.4f}",
                    flush=True,
                )

        self.pretraining_history_ = hist
        transferred, _ = transfer_torch_backbone_weights(src_net, self.model_, exclude_prefixes=("classifier",))
        self.transferred_layer_names_ = transferred

    def _run_masked_ssl_pretrain(
        self,
        X_fit: pd.DataFrame,
        y: Any,
        branch_specs: Mapping[str, Tuple[int, int]],
        static_dim: int,
        rng: np.random.Generator,
        pre_epochs: int,
    ) -> None:
        X_src = select_pretraining_sequences(
            X_fit, y, source=self.pretraining_source, sequence_col=self.extractor_.sequence_col,
        )
        eff_ratio = float(self.ssl_mask_ratio) if float(self.ssl_mask_ratio) > 0.0 else 0.25
        eff_span = max(1, int(self.ssl_span_length) if int(self.ssl_span_length) > 0 else 10)
        clean_chunks = self.extractor_.transform_chunks(X_src)

        src_net = self._build_net(branch_specs, static_dim, max(2, len(self.classes_)))
        dec_head = MaskedSSLReconstructionHead(
            branch_specs,
            branch_units=max(8, int(self.branch_units)),
            fusion_units=max(8, int(self.fusion_units)),
            decoder_units=int(self.ssl_decoder_units),
            decoder_dropout=float(self.ssl_decoder_dropout),
        ).to(self.device_)

        params = list(src_net.parameters()) + list(dec_head.parameters())
        opt = torch.optim.AdamW(params, lr=self._effective_pretraining_lr(), weight_decay=float(self.weight_decay))
        bs = self._effective_pretraining_batch_size()
        hist: Dict[str, List[float]] = {"loss": []}

        cached_pair: Optional[Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, np.ndarray]]] = None
        seed_base = int(self.random_state or 42)

        for epoch in range(1, pre_epochs + 1):
            if cached_pair is None or bool(self.ssl_resample_mask_each_epoch):
                corrupted_df = corrupt_raw_dataframe_spans(
                    X_src,
                    sequence_col=self.extractor_.sequence_col,
                    mask_ratio=eff_ratio,
                    span_length=eff_span,
                    mask_modalities_together=bool(self.ssl_mask_modalities_together),
                    seed=seed_base + epoch,
                )
                corrupt_chunks = self.extractor_.transform_chunks(corrupted_df)
                h_weights, t_clean = build_ssl_mask_weights(
                    corrupt_chunks, clean_chunks,
                    mask_ratio=eff_ratio, span_length=eff_span, seed=seed_base + epoch,
                )
                cached_pair = (corrupt_chunks, h_weights, t_clean)

            corrupt_chunks, h_weights, t_clean = cached_pair
            n_samples = len(corrupt_chunks["sequence_ids"])
            order = rng.permutation(n_samples)
            src_net.train()
            dec_head.train()
            ep_loss = 0.0
            ep_tot = 0

            for start in range(0, n_samples, bs):
                idx = order[start:start + bs]
                if len(idx) == 1 and n_samples > 1:
                    idx = np.r_[idx, order[0]]
                b_br, b_mk, b_pr, b_st = self._slice_batch(corrupt_chunks, idx)
                b_tgt = {k: torch.as_tensor(t_clean[k][idx], dtype=torch.float32, device=self.device_) for k in branch_specs}
                b_w = {k: torch.as_tensor(h_weights[k][idx], dtype=torch.float32, device=self.device_) for k in branch_specs}

                opt.zero_grad(set_to_none=True)
                branch_embs, fused = src_net.encode_with_branches(b_br, b_mk, b_pr, b_st)
                recons = dec_head(branch_embs, fused)
                loss = _masked_ssl_reconstruction_loss(recons, b_tgt, b_w, loss_name=self.ssl_loss_name)
                loss.backward()
                if self.gradient_clip_norm is not None and float(self.gradient_clip_norm) > 0:
                    nn.utils.clip_grad_norm_(params, float(self.gradient_clip_norm))
                opt.step()
                ep_loss += float(loss.detach().cpu().item()) * len(idx)
                ep_tot += len(idx)

            hist["loss"].append(ep_loss / max(1, ep_tot))
            if self.verbose and int(self.verbose) > 0:
                print(f"  [masked-ssl {epoch}/{pre_epochs}] recon_loss={hist['loss'][-1]:.4f}", flush=True)

        self.pretraining_history_ = hist
        transferred, _ = transfer_torch_backbone_weights(src_net, self.model_, exclude_prefixes=("classifier",))
        self.transferred_layer_names_ = transferred

    def _run_head_warmup(
        self,
        train_chunks: Mapping[str, Any],
        y_train_win: np.ndarray,
        rng: np.random.Generator,
        criterion: nn.Module,
        head_epochs: int,
        *,
        val_chunks: Optional[Mapping[str, Any]] = None,
        y_val_win: Optional[np.ndarray] = None,
    ) -> None:
        """Freeze backbone/fusion layers and train only ``self.model_.classifier`` for ``head_epochs``."""
        orig_req = {p_name: p.requires_grad for p_name, p in self.model_.named_parameters()}
        for p_name, p in self.model_.named_parameters():
            p.requires_grad = (p_name == "classifier" or p_name.startswith("classifier."))

        head_params = [p for p in self.model_.parameters() if p.requires_grad]
        head_lr = float(self.finetuning_head_learning_rate)
        eff_lr = head_lr if head_lr > 0.0 else float(self.learning_rate)
        head_opt = torch.optim.AdamW(head_params, lr=eff_lr, weight_decay=float(self.weight_decay))

        n_samples = len(y_train_win)
        bs = max(1, int(self.batch_size))
        hist: Dict[str, List[float]] = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

        try:
            for epoch in range(1, head_epochs + 1):
                self.model_.train()
                # Keep frozen BatchNorm running statistics steady while warming up the new head
                for m_name, mod in self.model_.named_modules():
                    if not (m_name == "classifier" or m_name.startswith("classifier.")) and isinstance(
                        mod, (nn.BatchNorm1d, nn.BatchNorm2d)
                    ):
                        mod.eval()
                order = rng.permutation(n_samples)
                ep_loss = 0.0
                corr = 0
                tot = 0
                for start in range(0, n_samples, bs):
                    idx = order[start:start + bs]
                    if len(idx) == 1 and n_samples > 1:
                        idx = np.r_[idx, order[0]]
                    b_br, b_mk, b_pr, b_st = self._slice_batch(train_chunks, idx)
                    b_y = torch.as_tensor(y_train_win[idx], dtype=torch.long, device=self.device_)
                    head_opt.zero_grad(set_to_none=True)
                    logits = self.model_(b_br, b_mk, b_pr, b_st)
                    loss = criterion(logits, b_y)
                    loss.backward()
                    if self.gradient_clip_norm is not None and float(self.gradient_clip_norm) > 0:
                        nn.utils.clip_grad_norm_(head_params, float(self.gradient_clip_norm))
                    head_opt.step()
                    ep_loss += float(loss.detach().cpu().item()) * len(idx)
                    corr += int((logits.detach().argmax(dim=-1) == b_y).sum().cpu().item())
                    tot += len(idx)

                tr_loss = ep_loss / max(1, tot)
                tr_acc = corr / max(1, tot)
                if val_chunks is not None and y_val_win is not None and len(y_val_win):
                    v_loss, v_acc = self._evaluate_chunks(val_chunks, y_val_win, criterion)
                else:
                    v_loss, v_acc = tr_loss, tr_acc
                hist["loss"].append(tr_loss)
                hist["accuracy"].append(tr_acc)
                hist["val_loss"].append(v_loss)
                hist["val_accuracy"].append(v_acc)
                if self.verbose and int(self.verbose) > 0:
                    print(
                        f"  [head-warmup {epoch}/{head_epochs}] loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                        f"val_loss={v_loss:.4f}",
                        flush=True,
                    )
        finally:
            for p_name, p in self.model_.named_parameters():
                p.requires_grad = orig_req[p_name]

        self.finetuning_head_history_ = hist


class PretrainedMultiBranchTSAIClassifier:
    """Factory/tag alias signaling pretraining-enabled ``MultiBranchTSAIClassifier`` usage."""

    def __new__(cls, *args, **kwargs):
        try:
            from .utils import MultiBranchTSAIClassifier
        except ImportError:
            from utils import MultiBranchTSAIClassifier
        kwargs.setdefault("pretraining_mode", "supervised")
        return MultiBranchTSAIClassifier(*args, **kwargs)


__all__ = [
    "PRETRAINING_MODES",
    "PRETRAINING_SOURCES",
    "SSL_LOSSES",
    "select_pretraining_sequences",
    "transfer_torch_backbone_weights",
    "MaskedSSLReconstructionHead",
    "corrupt_raw_dataframe_spans",
    "build_ssl_mask_weights",
    "PretrainingAndSSLMixin",
    "PretrainedMultiBranchTSAIClassifier",
]
