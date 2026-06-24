import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

try:
    from base_utils_qwen import competition_score
except ImportError:
    competition_score = None
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

def supervised_contrastive_loss(labels, embeddings, temperature=0.1):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).
    Pulls together embeddings of the same class, pushes apart different classes.
    """
    batch_size = tf.shape(labels)[0]
    labels = tf.cast(labels, tf.int32)
    labels = tf.reshape(labels, [-1, 1])
    
    # Cosine similarity via dot product of L2-normalized vectors
    sim_matrix = tf.matmul(embeddings, embeddings, transpose_b=True) / temperature
    mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)
    logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
    mask = mask * logits_mask
    
    # Numerical stability
    logits_max = tf.reduce_max(sim_matrix, axis=1, keepdims=True)
    logits = sim_matrix - logits_max
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-12)
    
    positives_count = tf.reduce_sum(mask, axis=1)
    valid_samples = tf.cast(positives_count > 0, tf.float32)
    
    mean_log_prob = tf.reduce_sum(mask * log_prob, axis=1) / (positives_count + 1e-12)
    loss = -tf.reduce_sum(mean_log_prob * valid_samples) / (tf.reduce_sum(valid_samples) + 1e-12)
    return loss

class ContrastiveSiameseModel(keras.Model):
    def __init__(self, backbone, projector, classifier, temperature=0.1, contrastive_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.projector = projector
        self.classifier = classifier
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        
        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.class_loss_tracker = keras.metrics.Mean(name="class_loss")
        self.accuracy = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        
    @property
    def metrics(self):
        return [self.contrastive_loss_tracker, self.class_loss_tracker, self.accuracy]
        
    def call(self, inputs, training=False):
        features = self.backbone(inputs, training=training)
        embeddings = self.projector(features, training=training)
        embeddings_norm = tf.math.l2_normalize(embeddings, axis=1)
        logits = self.classifier(features, training=training)
        return embeddings_norm, logits

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            embeddings_norm, logits = self(x, training=True)
            c_loss = supervised_contrastive_loss(y, embeddings_norm, self.temperature)
            class_loss = keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
            class_loss = tf.reduce_mean(class_loss)
            total_loss = self.contrastive_weight * c_loss + (1.0 - self.contrastive_weight) * class_loss
            
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.contrastive_loss_tracker.update_state(c_loss)
        self.class_loss_tracker.update_state(class_loss)
        self.accuracy.update_state(y, logits)
        
        return {"loss": total_loss, "c_loss": c_loss, "class_loss": class_loss, "accuracy": self.accuracy.result()}

    def test_step(self, data):
        x, y = data
        embeddings_norm, logits = self(x, training=False)
        c_loss = supervised_contrastive_loss(y, embeddings_norm, self.temperature)
        class_loss = keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
        class_loss = tf.reduce_mean(class_loss)
        total_loss = self.contrastive_weight * c_loss + (1.0 - self.contrastive_weight) * class_loss
        
        self.contrastive_loss_tracker.update_state(c_loss)
        self.class_loss_tracker.update_state(class_loss)
        self.accuracy.update_state(y, logits)
        return {"loss": total_loss, "c_loss": c_loss, "class_loss": class_loss, "accuracy": self.accuracy.result()}


class KerasContrastiveSiameseClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"
    
    def __init__(self,
                 target="bfrb",
                 maxlen=128,
                 padding_value=0.0,  # 0.0 is required for Keras Masking layer
                 backbone_filters="64-128",
                 kernel_sizes="3-3",
                 temporal_mode="attention",  # 'gru', 'attention', 'pool'
                 embedding_dim=128,
                 contrastive_weight=0.5,
                 temperature=0.1,
                 dense_units="64",
                 dropout=0.3,
                 learning_rate=1e-3,
                 batch_size=32,
                 epochs=50,
                 patience=10,
                 verbose=0,
                 random_state=42):
        self.target = target
        self.maxlen = maxlen
        self.padding_value = padding_value
        self.backbone_filters = backbone_filters
        self.kernel_sizes = kernel_sizes
        self.temporal_mode = temporal_mode
        self.embedding_dim = embedding_dim
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.dense_units = dense_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.random_state = random_state

    def _to_tuple(self, value):
        if value is None: return ()
        if isinstance(value, str):
            if value == "none": return ()
            return tuple(None if p == "none" else int(p) for p in value.split("-"))
        if isinstance(value, (tuple, list)): return tuple(value)
        return (value,)

    def _align(self, value, n: int):
        value = self._to_tuple(value)
        if len(value) == 0: return (None,) * n
        if len(value) == n: return value
        if len(value) == 1: return value * n
        if len(value) < n: return value + (value[-1],) * (n - len(value))
        return value[:n]

    def _unpack_X(self, X):
        """Handles dict output from base_utils_qwen SequenceExtractor"""
        if isinstance(X, dict):
            return X['X'], X.get('sequence_ids')
        return X, None

    def _collapse_y(self, seq_ids, y):
        if isinstance(y, pd.DataFrame):
            if "sequence_id" not in y.columns:
                raise ValueError("y dataframe must contain sequence_id.")
            if self.target not in y.columns:
                raise ValueError(f"y dataframe must contain target column: {self.target}")
            target_map = y.drop_duplicates("sequence_id").set_index("sequence_id")[self.target]
            if seq_ids is not None:
                y_seq = pd.Series(seq_ids).map(target_map)
            else:
                y_seq = y[self.target]
        else:
            y_seq = pd.Series(y).reset_index(drop=True)
        if y_seq.isna().any():
            raise ValueError("Missing labels")
        return y_seq.reset_index(drop=True)

    def _build(self, n_features, n_classes, maxlen=None):
        tf.keras.backend.clear_session()
        keras.utils.set_random_seed(self.random_state)

        seq_len = int(maxlen if maxlen is not None else self.maxlen)
        inp = keras.Input(shape=(seq_len, n_features))
        # Masking layer ensures RNNs and Attention ignore padded timesteps!
        x = layers.Masking(mask_value=self.padding_value)(inp)
        
        filters = self._to_tuple(self.backbone_filters)
        kernels = self._align(self.kernel_sizes, len(filters))
        for f, k in zip(filters, kernels):
            x = layers.Conv1D(f, k, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.SpatialDropout1D(0.1)(x)
            
        if self.temporal_mode == 'gru':
            x = layers.Bidirectional(layers.GRU(64, return_sequences=False))(x)
        elif self.temporal_mode == 'attention':
            attn = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
            x = layers.Add()([x, attn])
            x = layers.LayerNormalization()(x)
            ffn = layers.Dense(64, activation='relu')(x)
            ffn = layers.Dense(x.shape[-1])(ffn)
            x = layers.Add()([x, ffn])
            x = layers.LayerNormalization()(x)
            x = layers.GlobalAveragePooling1D()(x)
        else:
            x = layers.GlobalAveragePooling1D()(x)
        backbone = keras.Model(inp, x, name="backbone")
        
        proj_inp = keras.Input(shape=(x.shape[-1],))
        px = layers.Dense(self.embedding_dim, activation='relu')(proj_inp)
        px = layers.Dense(self.embedding_dim)(px)
        projector = keras.Model(proj_inp, px, name="projector")
        
        cls_inp = keras.Input(shape=(x.shape[-1],))
        cx = cls_inp
        dense_units = self._to_tuple(self.dense_units)
        for u in dense_units:
            cx = layers.Dense(u, activation='relu')(cx)
            cx = layers.Dropout(self.dropout)(cx)
        out = layers.Dense(n_classes)(cx)
        classifier = keras.Model(cls_inp, out, name="classifier")
        
        model = ContrastiveSiameseModel(
            backbone, projector, classifier, 
            temperature=self.temperature, 
            contrastive_weight=self.contrastive_weight
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def fit(self, X, y):
        X_pad, seq_ids = self._unpack_X(X)
        y_seq = self._collapse_y(seq_ids, y)
        
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y_seq)
        self.classes_ = self.le_.classes_
        
        self.maxlen_ = X_pad.shape[1]
        self.model_ = self._build(X_pad.shape[2], len(self.classes_), maxlen=self.maxlen_)
        
        self.history_ = self.model_.fit(
            X_pad, y_enc,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.15,
            callbacks=[keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True, monitor='val_loss')],
            verbose=self.verbose
        )
        return self

    def predict_proba(self, X):
        X_pad, _ = self._unpack_X(X)
        _, logits = self.model_.predict(X_pad, verbose=0)
        return tf.nn.softmax(logits).numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.le_.inverse_transform(np.argmax(proba, axis=1))

    def score(self, X, y):
        preds = self.predict(X)
        if isinstance(y, pd.DataFrame) and "sequence_id" in y.columns:
            y_seq = y.drop_duplicates("sequence_id").sort_values("sequence_id")
            if self.target == "bfrb" and "is_target" in y_seq.columns and competition_score is not None:
                return competition_score(
                    y_seq[self.target].values,
                    preds,
                    y_true_binary=y_seq["is_target"].astype(int).values,
                    target_only_macro=True,
                )
            return f1_score(y_seq[self.target].values, preds, average="macro", zero_division=0)
        y_seq = self._collapse_y(None, y)
        return f1_score(y_seq.to_numpy(), preds, average="macro", zero_division=0)