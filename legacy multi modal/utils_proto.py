from __future__ import annotations

from typing import Literal, Optional, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import keras
from tensorflow.keras import layers
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore", ".*mask.*Conv1D.*")


# ---------------------------------------------------------------------------
# Task sampler
# ---------------------------------------------------------------------------

class EpisodicTaskSampler:
    """Sample N-way K-shot episodes from a pool of (sequence_id, label) pairs.

    Parameters
    ----------
    n_way : int
        Number of classes per episode.
    k_shot : int
        Number of support examples per class.
    n_query : int
        Number of query examples per class.
    rng : np.random.Generator, optional
        Random number generator.  Created from ``random_state`` when None.
    random_state : int
        Seed used when ``rng`` is None.
    """

    def __init__(
        self,
        n_way: int = 4,
        k_shot: int = 5,
        n_query: int = 5,
        rng: Optional[np.random.Generator] = None,
        random_state: int = 42,
    ) -> None:
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.random_state = random_state
        self._rng = rng or np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    def fit(
        self,
        seq_ids: pd.Series,
        labels: pd.Series,
    ) -> "EpisodicTaskSampler":
        """Index (sequence_id → encoded label) for fast episode sampling."""
        assert len(seq_ids) == len(labels), "seq_ids and labels must be same length."
        self.seq_ids_ = np.asarray(seq_ids)
        self.labels_ = np.asarray(labels)
        self.classes_ = np.unique(self.labels_)
        self.class_to_indices_: dict[int, np.ndarray] = {
            c: np.flatnonzero(self.labels_ == c) for c in self.classes_
        }
        return self

    # ------------------------------------------------------------------
    def sample_episode(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (support_ids, support_labels, query_ids, query_labels).

        Labels are *local* 0..N-1 within the episode so the loss is always
        over ``n_way`` classes regardless of total number of classes.
        """
        available = [
            c for c in self.classes_
            if len(self.class_to_indices_[c]) >= self.k_shot + self.n_query
        ]
        if len(available) < self.n_way:
            # Fall back: allow classes with fewer examples, sampling with replacement
            available = list(self.classes_)

        episode_classes = self._rng.choice(available, size=self.n_way, replace=False)

        support_ids, support_labels = [], []
        query_ids, query_labels = [], []

        for local_label, cls in enumerate(episode_classes):
            pool = self.class_to_indices_[cls].copy()
            self._rng.shuffle(pool)
            n_needed = self.k_shot + self.n_query
            if len(pool) < n_needed:
                pool = self._rng.choice(pool, size=n_needed, replace=True)
            support_pool = pool[: self.k_shot]
            query_pool = pool[self.k_shot : self.k_shot + self.n_query]

            support_ids.extend(self.seq_ids_[support_pool])
            support_labels.extend([local_label] * self.k_shot)
            query_ids.extend(self.seq_ids_[query_pool])
            query_labels.extend([local_label] * self.n_query)

        return (
            np.array(support_ids),
            np.array(support_labels, dtype=np.int32),
            np.array(query_ids),
            np.array(query_labels, dtype=np.int32),
        )


# ---------------------------------------------------------------------------
# Encoder builder  (mirrors KerasMultiBranchClassifier._build but no head)
# ---------------------------------------------------------------------------

def _to_tuple(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, str):
        if value == "none":
            return ()
        return tuple(None if p == "none" else int(p) for p in value.split("-"))
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _align(value: Any, n: int) -> tuple:
    value = _to_tuple(value)
    if len(value) == 0:
        return (None,) * n
    if len(value) == n:
        return value
    if len(value) == 1:
        return value * n
    if len(value) < n:
        return value + (value[-1],) * (n - len(value))
    return value[:n]


def build_encoder(
    maxlen: int,
    n_features: int,
    branch_order: list[str],
    branch_indices: dict[str, list[int]],
    branch_filters: dict,
    branch_kernel_sizes: dict,
    branch_pool_sizes: dict,
    fusion_mode: str = "attention",
    attention_heads: int = 4,
    gru_units: int = 128,
    use_batch_norm: bool = True,
    spatial_dropout: float = 0.1,
    embed_dim: int = 128,
    dropout: float = 0.2,
) -> keras.Model:
    """Build a multi-branch 1D CNN encoder that outputs a fixed-size embedding.

    The architecture is identical to ``KerasMultiBranchClassifier._build`` up
    to (and including) the global pooling step.  Instead of a softmax head,
    a final ``Dense(embed_dim)`` with L2 normalisation is appended so that
    Euclidean distance in embedding space is well-conditioned.

    Parameters
    ----------
    maxlen : int
        Padded sequence length.
    n_features : int
        Number of input feature channels.
    branch_order : list[str]
        Ordered list of branch names (e.g. ["acc", "rot", "tof", "thm"]).
    branch_indices : dict[str, list[int]]
        Maps branch name → feature column indices.
    branch_filters, branch_kernel_sizes, branch_pool_sizes : dict
        Per-branch architecture strings (hyphen-separated integers).
    fusion_mode : {"attention", "bigru", "none"}
    attention_heads, gru_units : int
    use_batch_norm : bool
    spatial_dropout : float
    embed_dim : int
        Dimensionality of the output embedding.
    dropout : float
        Dropout before the embedding projection.

    Returns
    -------
    keras.Model
        Model with input shape ``(batch, maxlen, n_features)`` and output
        shape ``(batch, embed_dim)``.
    """
    inp = keras.Input(shape=(maxlen, n_features), name="input")

    branch_outs = []
    for br_name in branch_order:
        idxs = branch_indices[br_name]
        x = layers.Lambda(
            lambda t, i=idxs: tf.gather(t, i, axis=-1),
            name=f"branch_{br_name}_slice",
        )(inp)

        filters = _to_tuple(branch_filters.get(br_name, "32"))
        if len(filters) == 0:
            filters = (32,)
        n_layers = len(filters)
        kernels = _align(branch_kernel_sizes.get(br_name, "3"), n_layers)
        pools = _align(branch_pool_sizes.get(br_name, "none"), n_layers)

        for f, k, p in zip(filters, kernels, pools):
            x = layers.Conv1D(int(f), int(k), padding="same", activation="relu")(x)
            if use_batch_norm:
                x = layers.BatchNormalization()(x)
            if float(spatial_dropout) > 0:
                x = layers.SpatialDropout1D(float(spatial_dropout))(x)
            if p is not None:
                x = layers.MaxPooling1D(pool_size=int(p))(x)

        branch_outs.append(x)

    # Concatenation + fusion
    if len(branch_outs) == 1:
        fused = branch_outs[0]
        already_pooled = False
    else:
        time_dims = [b.shape[1] for b in branch_outs]
        if len(set(time_dims)) > 1:
            branch_outs = [
                layers.GlobalAveragePooling1D(name=f"branch_pool_{n}")(b)
                for n, b in zip(branch_order, branch_outs)
            ]
            fused = layers.Concatenate(axis=-1, name="branch_concat")(branch_outs)
            already_pooled = True
        else:
            fused = layers.Concatenate(axis=-1, name="branch_concat")(branch_outs)
            already_pooled = False

    if already_pooled:
        if fusion_mode == "attention":
            reshaped = layers.Reshape((1, fused.shape[-1]))(fused)
            attn = layers.MultiHeadAttention(
                num_heads=int(attention_heads),
                key_dim=max(1, int(fused.shape[-1]) // int(attention_heads)),
                name="fusion_attn",
            )(reshaped, reshaped)
            x = layers.Add()([reshaped, attn])
            x = layers.LayerNormalization()(x)
            x = layers.Flatten()(x)
        elif fusion_mode == "bigru":
            tokens = [layers.Reshape((1, b.shape[-1]))(b) for b in branch_outs]
            x = layers.Concatenate(axis=1, name="branch_tokens")(tokens)
            x = layers.Bidirectional(
                layers.GRU(int(gru_units), return_sequences=False),
                name="fusion_bigru",
            )(x)
        else:
            x = fused
    else:
        if fusion_mode == "attention":
            attn = layers.MultiHeadAttention(
                num_heads=int(attention_heads),
                key_dim=max(1, int(fused.shape[-1]) // int(attention_heads)),
                name="fusion_attn",
            )(fused, fused)
            x = layers.Add()([fused, attn])
            x = layers.LayerNormalization()(x)
        elif fusion_mode == "bigru":
            x = layers.Bidirectional(
                layers.GRU(int(gru_units), return_sequences=True),
                name="fusion_bigru",
            )(fused)
        else:
            x = fused
        x = layers.GlobalAveragePooling1D(name="global_pool")(x)

    if float(dropout) > 0:
        x = layers.Dropout(float(dropout))(x)

    # Projection + L2 normalisation → embedding lives on hypersphere
    x = layers.Dense(int(embed_dim), activation="relu", name="pre_embed")(x)
    embedding = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=-1), name="embedding"
    )(x)

    return keras.Model(inp, embedding, name="encoder")


# ---------------------------------------------------------------------------
# Episodic loss
# ---------------------------------------------------------------------------

@tf.function
def prototypical_loss(
    support_embeddings: tf.Tensor,
    support_labels: tf.Tensor,
    query_embeddings: tf.Tensor,
    query_labels: tf.Tensor,
    n_way: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compute prototypical-network cross-entropy loss for one episode.

    Parameters
    ----------
    support_embeddings : shape (n_way * k_shot, embed_dim)
    support_labels     : shape (n_way * k_shot,) — local 0..n_way-1
    query_embeddings   : shape (n_way * n_query, embed_dim)
    query_labels       : shape (n_way * n_query,) — local 0..n_way-1
    n_way              : int

    Returns
    -------
    loss : scalar tensor
    acc  : scalar tensor (episode accuracy on query set)
    """
    # Compute one prototype per class by averaging support embeddings
    prototypes = tf.stack(
        [
            tf.reduce_mean(
                tf.boolean_mask(support_embeddings, tf.equal(support_labels, c)),
                axis=0,
            )
            for c in range(n_way)
        ],
        axis=0,
    )  # (n_way, embed_dim)

    # Squared Euclidean distances: (n_query_total, n_way)
    diffs = (
        tf.expand_dims(query_embeddings, 1)  # (Q, 1, D)
        - tf.expand_dims(prototypes, 0)      # (1, N, D)
    )
    distances = tf.reduce_sum(tf.square(diffs), axis=-1)  # (Q, N)

    # Softmax over *negative* distances (closer = higher probability)
    log_probs = tf.nn.log_softmax(-distances, axis=-1)   # (Q, N)

    # Negative log-likelihood
    query_labels_oh = tf.cast(query_labels, tf.int32)
    loss = -tf.reduce_mean(
        tf.gather(log_probs, query_labels_oh, batch_dims=1)
    )

    # Accuracy
    preds = tf.argmax(-distances, axis=-1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, query_labels_oh), tf.float32))

    return loss, acc


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

class KerasPrototypicalClassifier(ClassifierMixin, BaseEstimator):
    """Prototypical Networks for few-shot gesture classification.

    Implements the episodic meta-training loop described in Snell et al. 2017
    and covered in Stanford CS330 Lecture 6 (Non-Parametric Few-Shot Learning).

    The encoder is a multi-branch 1D CNN (same architecture as
    ``KerasMultiBranchClassifier``) without the classification head.  During
    ``fit`` the encoder is trained via episodic tasks sampled from the training
    set.  At inference time, prototypes are computed from a *support set*
    passed alongside the query sequences.

    Parameters
    ----------
    target : str
        Name of the target column in the y DataFrame.
    maxlen : int
        Sequence length after padding.
    padding_value : float
        Value used to pad shorter sequences.
    branch_config : dict, optional
        Maps branch name → list of column-name prefixes.  Defaults to the
        same config as ``KerasMultiBranchClassifier``.
    branch_filters, branch_kernel_sizes, branch_pool_sizes : dict, optional
        Per-branch Conv1D architecture strings.
    fusion_mode : {"attention", "bigru", "none"}
    attention_heads, gru_units : int
    use_batch_norm : bool
    spatial_dropout : float
    embed_dim : int
        Embedding dimensionality (output of encoder).
    dropout : float
        Dropout before the embedding projection layer.
    n_way : int
        Classes per episode during meta-training.  Set to the number of
        gesture classes in your dataset (e.g. 4) or lower for harder episodes.
    k_shot : int
        Support examples per class per episode.
    n_query : int
        Query examples per class per episode.
    episodes_per_epoch : int
        Number of episodes to sample each "epoch".
    meta_epochs : int
        Total number of meta-training epochs.
    patience : int
        Early stopping patience (episodes without val-accuracy improvement).
    learning_rate : float
    batch_episodes : int
        Number of episodes to accumulate gradients over before one update
        (gradient batching).  1 = standard per-episode updates.
    val_fraction : float
        Fraction of sequences held out for episodic validation.
    verbose : int
        0 = silent, 1 = epoch-level summary.
    random_state : int
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        target: str = "gesture_action",
        maxlen: int = 64,
        padding_value: float = -999.0,
        branch_config: Optional[dict] = None,
        branch_filters: Optional[dict] = None,
        branch_kernel_sizes: Optional[dict] = None,
        branch_pool_sizes: Optional[dict] = None,
        fusion_mode: str = "attention",
        attention_heads: int = 4,
        gru_units: int = 128,
        use_batch_norm: bool = True,
        spatial_dropout: float = 0.1,
        embed_dim: int = 128,
        dropout: float = 0.2,
        n_way: int = 4,
        k_shot: int = 5,
        n_query: int = 5,
        episodes_per_epoch: int = 200,
        meta_epochs: int = 50,
        patience: int = 10,
        learning_rate: float = 1e-3,
        batch_episodes: int = 4,
        val_fraction: float = 0.15,
        verbose: int = 1,
        random_state: int = 42,
    ) -> None:
        self.target = target
        self.maxlen = maxlen
        self.padding_value = padding_value
        self.branch_config = branch_config
        self.branch_filters = branch_filters
        self.branch_kernel_sizes = branch_kernel_sizes
        self.branch_pool_sizes = branch_pool_sizes
        self.fusion_mode = fusion_mode
        self.attention_heads = attention_heads
        self.gru_units = gru_units
        self.use_batch_norm = use_batch_norm
        self.spatial_dropout = spatial_dropout
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.episodes_per_epoch = episodes_per_epoch
        self.meta_epochs = meta_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.batch_episodes = batch_episodes
        self.val_fraction = val_fraction
        self.verbose = verbose
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Branch helpers (mirrors KerasMultiBranchClassifier)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_branch_config() -> dict:
        return {
            "acc": ["acc_", "lin_acc_"],
            "rot": ["rot_", "delta_rot_", "ang_vel_", "rot6d_"],
            "tof": ["tof_"],
            "thm": ["thm_"],
        }

    @staticmethod
    def _default_branch_filters() -> dict:
        return {"acc": "64-128", "rot": "32-64", "tof": "32", "thm": "16"}

    @staticmethod
    def _default_branch_kernel_sizes() -> dict:
        return {"acc": "3-3", "rot": "3-3", "tof": "3", "thm": "3"}

    @staticmethod
    def _default_branch_pool_sizes() -> dict:
        return {"acc": "none-none", "rot": "none-none", "tof": "none", "thm": "none"}

    def _get_branch_indices(
        self, all_columns: pd.Index
    ) -> tuple[list[str], dict[str, list[int]]]:
        config = self.branch_config or self._default_branch_config()
        branch_order: list[str] = []
        branch_indices: dict[str, list[int]] = {}
        remaining = set(range(len(all_columns)))
        for name, prefixes in config.items():
            idxs = [
                i
                for i, c in enumerate(all_columns)
                if any(str(c).startswith(p) for p in prefixes)
            ]
            if idxs:
                branch_order.append(name)
                branch_indices[name] = sorted(idxs)
                remaining -= set(idxs)
        if remaining:
            branch_order.append("other")
            branch_indices["other"] = sorted(remaining)
        return branch_order, branch_indices

    # ------------------------------------------------------------------
    # Padding (identical logic to KerasMultiBranchClassifier._pad)
    # ------------------------------------------------------------------

    def _pad(self, X: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
        grouped = list(X.groupby(level=0, sort=False))
        n_seq = len(grouped)
        n_feat = X.shape[1]
        out = np.full(
            (n_seq, int(self.maxlen), n_feat),
            float(self.padding_value),
            dtype=np.float32,
        )
        seq_ids = []
        for i, (sid, g) in enumerate(grouped):
            arr = g.to_numpy(dtype=np.float32)
            length = min(len(arr), int(self.maxlen))
            out[i, :length] = arr[:length]
            seq_ids.append(sid)
        return out, pd.Series(seq_ids, name="sequence_id")

    # ------------------------------------------------------------------
    # Collapse labels (identical to KerasMultiBranchClassifier._collapse_y)
    # ------------------------------------------------------------------

    def _collapse_y(self, seq_ids: pd.Series, y: Any) -> pd.Series:
        if isinstance(y, pd.DataFrame):
            if "sequence_id" not in y.columns:
                raise ValueError("y dataframe must contain sequence_id.")
            if self.target not in y.columns:
                raise ValueError(
                    f"y dataframe must contain target column: {self.target}"
                )
            target_map = (
                y.drop_duplicates("sequence_id")
                .set_index("sequence_id")[self.target]
            )
            y_seq = seq_ids.map(target_map)
        else:
            y_seq = pd.Series(y).reset_index(drop=True)
            if len(y_seq) != len(seq_ids):
                raise ValueError(
                    "If y is not a dataframe, it must already be one label per sequence."
                )
        if y_seq.isna().any():
            missing = seq_ids[y_seq.isna()].head(10).tolist()
            raise ValueError(f"Missing labels for sequence_ids: {missing}")
        return y_seq.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Build a lookup: sequence_id → padded tensor
    # ------------------------------------------------------------------

    def _build_lookup(
        self, X_pad: np.ndarray, seq_ids: pd.Series
    ) -> dict[Any, np.ndarray]:
        return {sid: X_pad[i] for i, sid in enumerate(seq_ids)}

    # ------------------------------------------------------------------
    # Gather embeddings for a list of sequence_ids
    # ------------------------------------------------------------------

    def _gather(
        self, lookup: dict, ids: np.ndarray
    ) -> np.ndarray:
        batch = np.stack([lookup[sid] for sid in ids], axis=0)
        return self.encoder_.predict(batch, verbose=0)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Any) -> "KerasPrototypicalClassifier":
        """Meta-train the encoder via episodic Prototypical Network training.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame indexed by sequence_id (row-per-timestep),
            as produced by ``SequenceExtractor.transform()``.
        y : pd.DataFrame or array-like
            If DataFrame: must contain ``sequence_id`` and ``self.target``.
            If array-like: one label per unique sequence_id in the same order
            as ``X.index.unique()``.

        Returns
        -------
        self
        """
        tf.keras.backend.clear_session()
        keras.utils.set_random_seed(int(self.random_state))

        # --- Pad sequences ------------------------------------------------
        self.branch_order_, self.branch_indices_ = self._get_branch_indices(X.columns)
        X_pad, seq_ids = self._pad(X)

        # --- Encode labels ------------------------------------------------
        y_seq = self._collapse_y(seq_ids, y)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y_seq)
        self.classes_ = self.le_.classes_

        # --- Train / val split (sequence level) ---------------------------
        rng = np.random.default_rng(int(self.random_state))
        n_seq = len(seq_ids)
        val_size = max(1, int(n_seq * float(self.val_fraction)))
        shuffled = rng.permutation(n_seq)
        val_idx = shuffled[:val_size]
        train_idx = shuffled[val_size:]

        train_seq_ids = seq_ids.iloc[train_idx].reset_index(drop=True)
        train_labels = pd.Series(y_enc[train_idx])
        val_seq_ids = seq_ids.iloc[val_idx].reset_index(drop=True)
        val_labels = pd.Series(y_enc[val_idx])

        # --- Build sequence lookup ----------------------------------------
        lookup = self._build_lookup(X_pad, seq_ids)

        # --- Task samplers ------------------------------------------------
        train_sampler = EpisodicTaskSampler(
            n_way=min(int(self.n_way), len(self.classes_)),
            k_shot=int(self.k_shot),
            n_query=int(self.n_query),
            random_state=int(self.random_state),
        ).fit(train_seq_ids, train_labels)

        val_sampler = EpisodicTaskSampler(
            n_way=min(int(self.n_way), len(self.classes_)),
            k_shot=int(self.k_shot),
            n_query=int(self.n_query),
            random_state=int(self.random_state) + 1,
        ).fit(val_seq_ids, val_labels)

        effective_n_way = train_sampler.n_way

        # --- Build encoder ------------------------------------------------
        self.encoder_ = build_encoder(
            maxlen=int(self.maxlen),
            n_features=X_pad.shape[2],
            branch_order=self.branch_order_,
            branch_indices=self.branch_indices_,
            branch_filters=self.branch_filters or self._default_branch_filters(),
            branch_kernel_sizes=self.branch_kernel_sizes or self._default_branch_kernel_sizes(),
            branch_pool_sizes=self.branch_pool_sizes or self._default_branch_pool_sizes(),
            fusion_mode=str(self.fusion_mode),
            attention_heads=int(self.attention_heads),
            gru_units=int(self.gru_units),
            use_batch_norm=bool(self.use_batch_norm),
            spatial_dropout=float(self.spatial_dropout),
            embed_dim=int(self.embed_dim),
            dropout=float(self.dropout),
        )

        optimiser = keras.optimizers.Adam(learning_rate=float(self.learning_rate))

        # --- Meta-training loop -------------------------------------------
        best_val_acc = -1.0
        best_weights: Optional[list] = None
        patience_counter = 0
        self.history_: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
        }

        for epoch in range(int(self.meta_epochs)):
            # -- Training episodes --
            train_losses, train_accs = [], []
            accumulated_grads: Optional[list[tf.Variable]] = None
            n_accumulated = 0

            for ep_idx in range(int(self.episodes_per_epoch)):
                sup_ids, sup_labels, qry_ids, qry_labels = train_sampler.sample_episode()

                sup_batch = np.stack([lookup[sid] for sid in sup_ids], axis=0)
                qry_batch = np.stack([lookup[sid] for sid in qry_ids], axis=0)

                sup_labels_t = tf.constant(sup_labels, dtype=tf.int32)
                qry_labels_t = tf.constant(qry_labels, dtype=tf.int32)

                with tf.GradientTape() as tape:
                    sup_emb = self.encoder_(sup_batch, training=True)
                    qry_emb = self.encoder_(qry_batch, training=True)
                    loss, acc = prototypical_loss(
                        sup_emb, sup_labels_t, qry_emb, qry_labels_t, effective_n_way
                    )
                    scaled_loss = loss / float(self.batch_episodes)

                grads = tape.gradient(scaled_loss, self.encoder_.trainable_variables)
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = [
                        ag + g for ag, g in zip(accumulated_grads, grads)
                    ]
                n_accumulated += 1

                if n_accumulated >= int(self.batch_episodes) or ep_idx == int(self.episodes_per_epoch) - 1:
                    optimiser.apply_gradients(
                        zip(accumulated_grads, self.encoder_.trainable_variables)
                    )
                    accumulated_grads = None
                    n_accumulated = 0

                train_losses.append(float(loss))
                train_accs.append(float(acc))

            # -- Validation episodes (50 fixed) --
            val_losses, val_accs = [], []
            n_val_ep = min(50, int(self.episodes_per_epoch) // 4)
            for _ in range(n_val_ep):
                sup_ids, sup_labels, qry_ids, qry_labels = val_sampler.sample_episode()
                sup_batch = np.stack([lookup[sid] for sid in sup_ids], axis=0)
                qry_batch = np.stack([lookup[sid] for sid in qry_ids], axis=0)
                sup_emb = self.encoder_(sup_batch, training=False)
                qry_emb = self.encoder_(qry_batch, training=False)
                loss_v, acc_v = prototypical_loss(
                    sup_emb,
                    tf.constant(sup_labels, dtype=tf.int32),
                    qry_emb,
                    tf.constant(qry_labels, dtype=tf.int32),
                    effective_n_way,
                )
                val_losses.append(float(loss_v))
                val_accs.append(float(acc_v))

            mean_train_loss = np.mean(train_losses)
            mean_train_acc = np.mean(train_accs)
            mean_val_loss = np.mean(val_losses)
            mean_val_acc = np.mean(val_accs)

            self.history_["train_loss"].append(mean_train_loss)
            self.history_["train_acc"].append(mean_train_acc)
            self.history_["val_loss"].append(mean_val_loss)
            self.history_["val_acc"].append(mean_val_acc)

            if int(self.verbose) > 0:
                print(
                    f"Epoch {epoch + 1:3d}/{self.meta_epochs}"
                    f"  train_loss={mean_train_loss:.4f}  train_acc={mean_train_acc:.4f}"
                    f"  val_loss={mean_val_loss:.4f}  val_acc={mean_val_acc:.4f}"
                )

            # Early stopping
            if mean_val_acc > best_val_acc:
                best_val_acc = mean_val_acc
                best_weights = self.encoder_.get_weights()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= int(self.patience):
                    if int(self.verbose) > 0:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

        if best_weights is not None:
            self.encoder_.set_weights(best_weights)

        self.best_val_acc_ = best_val_acc
        return self

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _embed_all(self, X: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
        """Return (embeddings, seq_ids) for every sequence in X."""
        check_is_fitted(self, ["encoder_", "le_", "classes_"])
        X_pad, seq_ids = self._pad(X)
        embeddings = self.encoder_.predict(X_pad, verbose=0)
        return embeddings, seq_ids

    def _compute_prototypes(
        self,
        embeddings: np.ndarray,
        labels_enc: np.ndarray,
    ) -> np.ndarray:
        """Average embeddings per class to get prototypes."""
        prototypes = np.stack(
            [
                embeddings[labels_enc == c].mean(axis=0)
                for c in range(len(self.classes_))
            ],
            axis=0,
        )
        # L2 normalise prototypes to keep them on the hypersphere
        norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return prototypes / norms

    # ------------------------------------------------------------------
    # predict_proba / predict (standard inference — fit() provides prototypes)
    # ------------------------------------------------------------------

    def fit_support(self, X_support: pd.DataFrame, y_support: Any) -> "KerasPrototypicalClassifier":
        """Compute and store prototypes from a held-out support set.

        Call this after ``fit()`` to set up in-context prototypes for a
        specific test distribution.  If you skip this, ``predict_proba``
        falls back to using all training sequences as the support set.

        Parameters
        ----------
        X_support : pd.DataFrame
            Feature DataFrame for the support set (same format as ``X`` in ``fit``).
        y_support : pd.DataFrame or array-like
            Labels for the support set.
        """
        check_is_fitted(self, ["encoder_", "le_", "classes_"])
        X_pad, seq_ids = self._pad(X_support)
        y_seq = self._collapse_y(seq_ids, y_support)
        y_enc = self.le_.transform(y_seq)
        embeddings = self.encoder_.predict(X_pad, verbose=0)
        self.prototypes_ = self._compute_prototypes(embeddings, y_enc)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities via nearest-prototype softmax.

        Requires ``fit_support()`` to have been called, or ``prototypes_``
        to have been set manually.

        Parameters
        ----------
        X : pd.DataFrame
            Query sequences.

        Returns
        -------
        np.ndarray of shape (n_sequences, n_classes)
        """
        check_is_fitted(self, ["encoder_", "prototypes_"])
        query_emb, _ = self._embed_all(X)
        # Squared Euclidean distances to each prototype
        diffs = query_emb[:, None, :] - self.prototypes_[None, :, :]  # (Q, N, D)
        distances = np.sum(diffs ** 2, axis=-1)  # (Q, N)
        # Softmax over negative distances
        log_probs = -distances - np.log(np.sum(np.exp(-distances + np.max(-distances, axis=1, keepdims=True)), axis=1, keepdims=True) + 1e-8)
        # Stable softmax
        neg_d = -distances
        neg_d -= neg_d.max(axis=1, keepdims=True)
        exp_d = np.exp(neg_d)
        probs = exp_d / exp_d.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for query sequences."""
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return self.le_.inverse_transform(pred_idx)

    def score(self, X: pd.DataFrame, y: Any) -> float:
        """Return macro-F1 on the query set (requires ``fit_support`` first)."""
        X_pad, seq_ids = self._pad(X)
        y_seq = self._collapse_y(seq_ids, y)
        preds = self.predict(X)
        return f1_score(y_seq.to_numpy(), preds, average="macro")
