
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DynamicMultiHeadPrototypicalNetwork(ClassifierMixin, BaseEstimator):
    """
    Dynamic multi-output Prototypical Network.
    Specify which heads you want: e.g., heads = ["gesture_action", "orientation", "phase"]
    Automatically builds lookup table from training data.
    """
    
    _estimator_type = "classifier"
    
    def __init__(
        self,
        heads=None,  # List of target columns: ["gesture_action", "orientation", "phase"]
        primary_target="gesture",  # Target for final F1 score (e.g., "gesture", "gesture_action")
        n_way=20,
        n_support=5,
        n_query=15,
        backbone_type="1dcnn",
        conv_filters="64",
        kernel_sizes="5",
        pool_sizes="none",
        use_batch_norm=True,
        spatial_dropout=0.1,
        dense_units="64",
        dropout=0.3,
        embedding_dim=128,
        distance="euclidean",
        learning_rate=1e-3,
        batch_size=32,
        epochs=100,
        patience=15,
        maxlen=160,
        padding_value=-999.0,
        verbose=0,
        random_state=42,
    ):
        self.heads = heads if heads is not None else ["gesture_action"]
        self.primary_target = primary_target
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.backbone_type = backbone_type
        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.use_batch_norm = use_batch_norm
        self.spatial_dropout = spatial_dropout
        self.dense_units = dense_units
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.distance = distance
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.maxlen = maxlen
        self.padding_value = padding_value
        self.verbose = verbose
        self.random_state = random_state

    def get_params(self, deep=True):
        return {
            'heads': self.heads,
            'primary_target': self.primary_target,
            'n_way': self.n_way,
            'n_support': self.n_support,
            'n_query': self.n_query,
            'backbone_type': self.backbone_type,
            'conv_filters': self.conv_filters,
            'kernel_sizes': self.kernel_sizes,
            'pool_sizes': self.pool_sizes,
            'use_batch_norm': self.use_batch_norm,
            'spatial_dropout': self.spatial_dropout,
            'dense_units': self.dense_units,
            'dropout': self.dropout,
            'embedding_dim': self.embedding_dim,
            'distance': self.distance,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
            'maxlen': self.maxlen,
            'padding_value': self.padding_value,
            'verbose': self.verbose,
            'random_state': self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _to_tuple(self, value):
        if value is None:
            return ()
        if isinstance(value, str):
            if value == "none":
                return ()
            return tuple(None if p == "none" else int(p) for p in value.split("-"))
        if isinstance(value, tuple):
            return value
        return (value,)

    def _align(self, value, n):
        value = self._to_tuple(value)
        if len(value) == 0:
            return (None,) * n
        if len(value) == n:
            return value
        if len(value) == 1:
            return value * n
        if len(value) < n:
            return value + (value[-1],) * (n - len(value))
        return value[:n]

    def _build_embedding_net(self, input_shape, head_dims):
        tf.keras.backend.clear_session()
        keras.utils.set_random_seed(self.random_state)
        
        inp = keras.Input(shape=input_shape)
        x = inp
        
        if self.backbone_type == "1dcnn":
            filters = self._to_tuple(self.conv_filters)
            n_layers = max(1, len(filters))
            kernels = self._align(self.kernel_sizes, n_layers)
            pools = self._align(self.pool_sizes, n_layers)
            
            for f, k, p in zip(filters, kernels, pools):
                x = layers.Conv1D(int(f), int(k), padding="same", activation="relu")(x)
                if self.use_batch_norm:
                    x = layers.BatchNormalization()(x)
                if self.spatial_dropout > 0:
                    x = layers.SpatialDropout1D(self.spatial_dropout)(x)
                if p is not None:
                    x = layers.MaxPooling1D(int(p))(x)
            x = layers.GlobalAveragePooling1D()(x)
        else:
            x = layers.Reshape((input_shape[0], input_shape[1], 1))(inp)
            filters = self._to_tuple(self.conv_filters)
            n_layers = max(1, len(filters))
            kernels = self._align(self.kernel_sizes, n_layers)
            pools = self._align(self.pool_sizes, n_layers)
            for f, k, p in zip(filters, kernels, pools):
                x = layers.Conv2D(int(f), (int(k), int(k)), padding="same", activation="relu")(x)
                if self.use_batch_norm:
                    x = layers.BatchNormalization()(x)
                if self.spatial_dropout > 0:
                    x = layers.SpatialDropout2D(self.spatial_dropout)(x)
                if p is not None:
                    x = layers.MaxPooling2D((int(p), int(p)))(x)
            x = layers.GlobalAveragePooling2D()(x)
        
        dense_units = self._to_tuple(self.dense_units)
        for units in dense_units:
            x = layers.Dense(int(units), activation="relu")(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)
        
        embedding = layers.Dense(self.embedding_dim, name="embedding")(x)
        
        outputs = []
        for head in self.heads:
            out = layers.Dense(head_dims[head], activation="softmax", name=head)(embedding)
            outputs.append(out)
        
        return keras.Model(inp, outputs)

    def _distance(self, z_proto, z_query):
        if self.distance == "euclidean":
            return tf.reduce_sum((z_proto[:, None] - z_query[None, :]) ** 2, axis=2)
        else:
            proto_norm = tf.nn.l2_normalize(z_proto, axis=-1)
            query_norm = tf.nn.l2_normalize(z_query, axis=-1)
            return 1.0 - tf.matmul(proto_norm, query_norm, transpose_b=True)

    def _pad(self, X):
        grouped = list(X.groupby(level=0, sort=False))
        n_seq = len(grouped)
        n_feat = X.shape[1]
        X_pad = np.full((n_seq, self.maxlen, n_feat), self.padding_value, dtype=np.float32)
        seq_order = []
        for i, (sid, g) in enumerate(grouped):
            arr = g.to_numpy(dtype=np.float32)
            length = min(len(arr), self.maxlen)
            X_pad[i, :length] = arr[:length]
            seq_order.append(sid)
        return X_pad, seq_order

    def _sample_episode(self, X, y_dict, classes_dict):
        # Pick random classes for each head independently
        episode_classes = {}
        for head in self.heads:
            episode_classes[head] = np.random.choice(classes_dict[head], size=self.n_way, replace=False)
        
        support_x, query_x = [], []
        support_y = {head: [] for head in self.heads}
        query_y = {head: [] for head in self.heads}
        
        for i in range(self.n_way):
            # Try to find indices where ALL heads match
            mask = np.ones(len(y_dict[self.heads[0]]), dtype=bool)
            for head in self.heads:
                mask &= (y_dict[head] == episode_classes[head][i])
            idx = np.where(mask)[0]
            
            # Fallback: use first head only if no match
            if len(idx) < self.n_support + self.n_query:
                mask = (y_dict[self.heads[0]] == episode_classes[self.heads[0]][i])
                idx = np.where(mask)[0]
            
            if len(idx) < self.n_support + self.n_query:
                idx = np.tile(idx, (self.n_support + self.n_query + len(idx) - 1) // len(idx))[:self.n_support + self.n_query]
            
            np.random.shuffle(idx)
            support_x.append(X[idx[:self.n_support]])
            for head in self.heads:
                support_y[head].extend([episode_classes[head][i]] * self.n_support)
            
            query_x.append(X[idx[self.n_support:self.n_support + self.n_query]])
            for head in self.heads:
                query_y[head].extend([episode_classes[head][i]] * self.n_query)
        
        support_x = np.concatenate(support_x)
        query_x = np.concatenate(query_x)
        for head in self.heads:
            support_y[head] = np.array(support_y[head])
            query_y[head] = np.array(query_y[head])
        
        return support_x, support_y, query_x, query_y, episode_classes

    def fit(self, X, y):
        # Process multi-output labels
        if isinstance(y, pd.DataFrame):
            y_encoded = {}
            classes_dict = {}
            self.label_encoders_ = {}
            
            for head in self.heads:
                if head not in y.columns:
                    raise ValueError(f"Head '{head}' not found in y columns")
                y_head = y.drop_duplicates("sequence_id").set_index("sequence_id")[head]
                le = LabelEncoder()
                y_encoded[head] = le.fit_transform(y_head)
                self.label_encoders_[head] = le
                classes_dict[head] = le.classes_
        else:
            raise ValueError("y must be DataFrame for multi-output")
        
        seq_ids = X.index.unique()
        for head in self.heads:
            y_encoded[head] = y_encoded[head].reindex(seq_ids)
            if y_encoded[head].isna().any():
                raise ValueError(f"Missing labels for head: {head}")
        
        X_pad, seq_order = self._pad(X)
        for head in self.heads:
            y_encoded[head] = y_encoded[head].reindex(pd.Index(seq_order)).to_numpy()
        
        head_dims = {head: len(self.label_encoders_[head].classes_) for head in self.heads}
        self.embedding_net_ = self._build_embedding_net((self.maxlen, X.shape[1]), head_dims)
        
        losses = {head: 'sparse_categorical_crossentropy' for head in self.heads}
        self.embedding_net_.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss=losses)
        
        classes_dict = {head: np.arange(len(self.label_encoders_[head].classes_)) for head in self.heads}
        
        optimizer = keras.optimizers.Adam(self.learning_rate)
        steps = max(1, len(X_pad) // self.batch_size)
        best_loss = np.inf
        patience_counter = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for _ in range(steps):
                sup_x, sup_y, qry_x, qry_y, ep_classes = self._sample_episode(X_pad, y_encoded, classes_dict)
                
                with tf.GradientTape() as tape:
                    all_emb = self.embedding_net_(sup_x, training=True)
                    sup_emb = all_emb[-len(self.heads)]  # embedding is last output? Actually embedding is first
                    # Re-arrange: embedding_net outputs [embedding, head1, head2, ...]
                    # Need to get embedding correctly
                    sup_emb = self.embedding_net_.get_layer("embedding")(sup_x, training=True)
                    qry_emb = self.embedding_net_.get_layer("embedding")(qry_x, training=True)
                    
                    total_loss = 0.0
                    for head in self.heads:
                        ep_classes_head = ep_classes[head]
                        sup_y_head = sup_y[head]
                        qry_y_head = qry_y[head]
                        
                        prototypes = tf.stack([tf.reduce_mean(sup_emb[sup_y_head == c], axis=0) for c in ep_classes_head])
                        dist = self._distance(prototypes, qry_emb)
                        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=qry_y_head, logits=tf.transpose(-dist)
                        ))
                        total_loss += loss
                    
                    # Add supervised losses from heads
                    head_outputs = self.embedding_net_(sup_x, training=True)
                    for idx, head in enumerate(self.heads):
                        total_loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=sup_y[head], logits=head_outputs[idx]
                        ))
                
                grads = tape.gradient(total_loss, self.embedding_net_.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.embedding_net_.trainable_variables))
                epoch_loss += total_loss.numpy()
            
            epoch_loss /= steps
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, loss: {epoch_loss:.4f}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        # Store prototypes for each head
        all_emb = self.embedding_net_.get_layer("embedding")(X_pad, training=False).numpy()
        self.prototypes_ = {}
        for head in self.heads:
            self.prototypes_[head] = np.array([
                np.mean(all_emb[y_encoded[head] == c], axis=0) for c in range(len(self.label_encoders_[head].classes_))
            ])
        
        # Build lookup table for primary target
        if self.primary_target in train_df.columns:
            cols = [self.primary_target] + self.heads
            self.lookup_table_ = train_df[train_df["sequence_type"] == "Target"][cols].drop_duplicates().reset_index(drop=True)
        else:
            self.lookup_table_ = None
        
        return self

    def predict(self, X):
        check_is_fitted(self, ["embedding_net_", "label_encoders_", "prototypes_"])
        X_pad, _ = self._pad(X)
        emb = self.embedding_net_.get_layer("embedding")(X_pad, training=False).numpy()
        
        # Predict each head
        preds = {}
        for head in self.heads:
            dist = self._distance(self.prototypes_[head], emb)
            preds[head] = self.label_encoders_[head].inverse_transform(np.argmin(dist, axis=0))
        
        # If we have a lookup table, decode to primary target
        if self.lookup_table_ is not None and self.primary_target in self.lookup_table_.columns:
            decoded = []
            for i in range(len(preds[self.heads[0]])):
                # Build filter
                mask = True
                for head in self.heads:
                    mask &= (self.lookup_table_[head] == preds[head][i])
                matches = self.lookup_table_[mask]
                if len(matches) > 0:
                    decoded.append(matches.iloc[0][self.primary_target])
                else:
                    # Fallback: concatenate
                    decoded.append(" | ".join([f"{preds[head][i]}" for head in self.heads]))
            return np.array(decoded)
        
        # Otherwise return first head
        return preds[self.heads[0]]

    def predict_all(self, X):
        """Return all head predictions"""
        check_is_fitted(self, ["embedding_net_", "label_encoders_", "prototypes_"])
        X_pad, _ = self._pad(X)
        emb = self.embedding_net_.get_layer("embedding")(X_pad, training=False).numpy()
        
        result = {}
        for head in self.heads:
            dist = self._distance(self.prototypes_[head], emb)
            result[head] = self.label_encoders_[head].inverse_transform(np.argmin(dist, axis=0))
        return result

    def score(self, X, y):
        from sklearn.metrics import f1_score
        y_pred = self.predict(X)
        if isinstance(y, pd.DataFrame):
            if self.primary_target in y.columns:
                y_true = y.drop_duplicates("sequence_id").set_index("sequence_id")[self.primary_target]
            else:
                y_true = y.drop_duplicates("sequence_id").set_index("sequence_id")[y.columns[0]]
            y_true = y_true.reindex(X.index.unique())
        else:
            y_true = pd.Series(y)
        return f1_score(y_true, y_pred, average="macro")
    

class BinaryPlusGesturePrototypicalNetwork(ClassifierMixin, BaseEstimator):
    """
    Prototypical Network with two heads:
    - Binary: target (1) vs non-target (0)
    - Gesture: pinch skin, pull hair, pull hairline, scratch (only for target sequences)
    
    Competition metric = (binary_f1 + gesture_macro_f1) / 2
    """
    
    _estimator_type = "classifier"
    
    def __init__(
        self,
        gesture_column="gesture",
        n_way=20,
        n_support=5,
        n_query=15,
        backbone_type="1dcnn",
        conv_filters="64",
        kernel_sizes="5",
        pool_sizes="none",
        use_batch_norm=True,
        spatial_dropout=0.1,
        dense_units="64",
        dropout=0.3,
        embedding_dim=128,
        distance="euclidean",
        learning_rate=1e-3,
        batch_size=32,
        epochs=100,
        patience=15,
        maxlen=160,
        padding_value=-999.0,
        verbose=0,
        random_state=42,
    ):
        self.gesture_column = gesture_column
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.backbone_type = backbone_type
        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.use_batch_norm = use_batch_norm
        self.spatial_dropout = spatial_dropout
        self.dense_units = dense_units
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.distance = distance
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.maxlen = maxlen
        self.padding_value = padding_value
        self.verbose = verbose
        self.random_state = random_state

    def get_params(self, deep=True):
        return {
            'gesture_column': self.gesture_column,
            'n_way': self.n_way,
            'n_support': self.n_support,
            'n_query': self.n_query,
            'backbone_type': self.backbone_type,
            'conv_filters': self.conv_filters,
            'kernel_sizes': self.kernel_sizes,
            'pool_sizes': self.pool_sizes,
            'use_batch_norm': self.use_batch_norm,
            'spatial_dropout': self.spatial_dropout,
            'dense_units': self.dense_units,
            'dropout': self.dropout,
            'embedding_dim': self.embedding_dim,
            'distance': self.distance,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
            'maxlen': self.maxlen,
            'padding_value': self.padding_value,
            'verbose': self.verbose,
            'random_state': self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _to_tuple(self, value):
        if value is None:
            return ()
        if isinstance(value, str):
            if value == "none":
                return ()
            return tuple(None if p == "none" else int(p) for p in value.split("-"))
        if isinstance(value, tuple):
            return value
        return (value,)

    def _align(self, value, n):
        value = self._to_tuple(value)
        if len(value) == 0:
            return (None,) * n
        if len(value) == n:
            return value
        if len(value) == 1:
            return value * n
        if len(value) < n:
            return value + (value[-1],) * (n - len(value))
        return value[:n]

    def _build_embedding_net(self, input_shape, n_gesture_classes):
        tf.keras.backend.clear_session()
        keras.utils.set_random_seed(self.random_state)
        
        inp = keras.Input(shape=input_shape)
        x = inp
        
        if self.backbone_type == "1dcnn":
            filters = self._to_tuple(self.conv_filters)
            n_layers = max(1, len(filters))
            kernels = self._align(self.kernel_sizes, n_layers)
            pools = self._align(self.pool_sizes, n_layers)
            
            for f, k, p in zip(filters, kernels, pools):
                x = layers.Conv1D(int(f), int(k), padding="same", activation="relu")(x)
                if self.use_batch_norm:
                    x = layers.BatchNormalization()(x)
                if self.spatial_dropout > 0:
                    x = layers.SpatialDropout1D(self.spatial_dropout)(x)
                if p is not None:
                    x = layers.MaxPooling1D(int(p))(x)
            x = layers.GlobalAveragePooling1D()(x)
        else:
            x = layers.Reshape((input_shape[0], input_shape[1], 1))(inp)
            filters = self._to_tuple(self.conv_filters)
            n_layers = max(1, len(filters))
            kernels = self._align(self.kernel_sizes, n_layers)
            pools = self._align(self.pool_sizes, n_layers)
            for f, k, p in zip(filters, kernels, pools):
                x = layers.Conv2D(int(f), (int(k), int(k)), padding="same", activation="relu")(x)
                if self.use_batch_norm:
                    x = layers.BatchNormalization()(x)
                if self.spatial_dropout > 0:
                    x = layers.SpatialDropout2D(self.spatial_dropout)(x)
                if p is not None:
                    x = layers.MaxPooling2D((int(p), int(p)))(x)
            x = layers.GlobalAveragePooling2D()(x)
        
        dense_units = self._to_tuple(self.dense_units)
        for units in dense_units:
            x = layers.Dense(int(units), activation="relu")(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)
        
        embedding = layers.Dense(self.embedding_dim, name="embedding")(x)
        
        binary_output = layers.Dense(1, activation="sigmoid", name="binary")(embedding)
        gesture_output = layers.Dense(n_gesture_classes, activation="softmax", name="gesture")(embedding)
        
        return keras.Model(inp, outputs=[binary_output, gesture_output])

    def _distance(self, z_proto, z_query):
        if self.distance == "euclidean":
            return tf.reduce_sum((z_proto[:, None] - z_query[None, :]) ** 2, axis=2)
        else:
            proto_norm = tf.nn.l2_normalize(z_proto, axis=-1)
            query_norm = tf.nn.l2_normalize(z_query, axis=-1)
            return 1.0 - tf.matmul(proto_norm, query_norm, transpose_b=True)

    def _pad(self, X):
        grouped = list(X.groupby(level=0, sort=False))
        n_seq = len(grouped)
        n_feat = X.shape[1]
        X_pad = np.full((n_seq, self.maxlen, n_feat), self.padding_value, dtype=np.float32)
        seq_order = []
        for i, (sid, g) in enumerate(grouped):
            arr = g.to_numpy(dtype=np.float32)
            length = min(len(arr), self.maxlen)
            X_pad[i, :length] = arr[:length]
            seq_order.append(sid)
        return X_pad, seq_order

    def _sample_episode(self, X, y_binary, y_gesture, gesture_classes):
        # Sample episodes using gesture classes only (binary is determined by gesture)
        episode_classes = np.random.choice(gesture_classes, size=self.n_way, replace=False)
        
        support_x, query_x = [], []
        support_binary, support_gesture = [], []
        query_binary, query_gesture = [], []
        
        for cls in episode_classes:
            idx = np.where(y_gesture == cls)[0]
            np.random.shuffle(idx)
            
            if len(idx) < self.n_support + self.n_query:
                idx = np.tile(idx, (self.n_support + self.n_query + len(idx) - 1) // len(idx))[:self.n_support + self.n_query]
            
            support_x.append(X[idx[:self.n_support]])
            support_binary.extend([1] * self.n_support)
            support_gesture.extend([cls] * self.n_support)
            
            query_x.append(X[idx[self.n_support:self.n_support + self.n_query]])
            query_binary.extend([1] * self.n_query)
            query_gesture.extend([cls] * self.n_query)
        
        # Also add non-target examples (binary=0)
        non_target_idx = np.where(y_binary == 0)[0]
        if len(non_target_idx) > 0:
            np.random.shuffle(non_target_idx)
            n_non_target = min(len(non_target_idx), self.n_support + self.n_query)
            non_target_sample = non_target_idx[:n_non_target]
            support_x.append(X[non_target_sample[:self.n_support]])
            support_binary.extend([0] * min(self.n_support, len(non_target_sample)))
            support_gesture.extend([-1] * min(self.n_support, len(non_target_sample)))
            
            if len(non_target_sample) > self.n_support:
                query_x.append(X[non_target_sample[self.n_support:self.n_support + self.n_query]])
                query_binary.extend([0] * min(self.n_query, len(non_target_sample) - self.n_support))
                query_gesture.extend([-1] * min(self.n_query, len(non_target_sample) - self.n_support))
        
        support_x = np.concatenate(support_x)
        query_x = np.concatenate(query_x)
        
        return (support_x, np.array(support_binary), np.array(support_gesture),
                query_x, np.array(query_binary), np.array(query_gesture), episode_classes)

    def fit(self, X, y):
        # y should have columns: sequence_id, is_target, gesture
        if isinstance(y, pd.DataFrame):
            y_binary = y.drop_duplicates("sequence_id").set_index("sequence_id")["is_target"]
            y_gesture = y.drop_duplicates("sequence_id").set_index("sequence_id")[self.gesture_column]
        else:
            raise ValueError("y must be DataFrame with 'is_target' and 'gesture' columns")
        
        seq_ids = X.index.unique()
        y_binary = y_binary.reindex(seq_ids)
        y_gesture = y_gesture.reindex(seq_ids)
        
        # Encode gesture labels (non-target gets -1, we'll handle separately)
        self.gesture_encoder_ = LabelEncoder()
        # Only fit on target gestures
        target_gestures = y_gesture[y_gesture != "non_target"].dropna().unique()
        self.gesture_encoder_.fit(target_gestures)
        self.gesture_classes_ = self.gesture_encoder_.classes_
        
        # Convert to numeric
        y_binary = y_binary.fillna(0).astype(int).to_numpy()
        y_gesture_encoded = np.full(len(y_gesture), -1, dtype=int)
        mask_target = y_gesture.isin(self.gesture_classes_)
        y_gesture_encoded[mask_target] = self.gesture_encoder_.transform(y_gesture[mask_target])
        
        X_pad, seq_order = self._pad(X)
        y_binary = y_binary[[seq_ids.get_loc(sid) for sid in seq_order]]
        y_gesture_encoded = y_gesture_encoded[[seq_ids.get_loc(sid) for sid in seq_order]]
        
        self.embedding_net_ = self._build_embedding_net((self.maxlen, X.shape[1]), len(self.gesture_classes_))
        
        # Create embedding extractor model
        self.embedding_extractor_ = keras.Model(
            inputs=self.embedding_net_.inputs,
            outputs=self.embedding_net_.get_layer("embedding").output
        )
        
        self.embedding_net_.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss={"binary": "binary_crossentropy", "gesture": "sparse_categorical_crossentropy"},
            loss_weights={"binary": 1.0, "gesture": 1.0}
        )
        
        # Only use target sequences for prototype computation
        target_mask = y_gesture_encoded >= 0
        X_target = X_pad[target_mask]
        y_binary_target = y_binary[target_mask]
        y_gesture_target = y_gesture_encoded[target_mask]
        
        gesture_classes = np.arange(len(self.gesture_classes_))
        
        optimizer = keras.optimizers.Adam(self.learning_rate)
        steps = max(1, len(X_target) // self.batch_size)
        best_loss = np.inf
        patience_counter = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for _ in range(steps):
                sup_x, sup_bin, sup_ges, qry_x, qry_bin, qry_ges, ep_classes = self._sample_episode(
                    X_target, y_binary_target, y_gesture_target, gesture_classes
                )
                
                with tf.GradientTape() as tape:
                    # Get embeddings using the extractor model
                    sup_emb = self.embedding_extractor_(sup_x, training=True)
                    qry_emb = self.embedding_extractor_(qry_x, training=True)
                    
                    # Prototype loss for gesture
                    prototypes = tf.stack([tf.reduce_mean(sup_emb[sup_ges == c], axis=0) for c in ep_classes])
                    dist = self._distance(prototypes, qry_emb)
                    gesture_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=qry_ges, logits=tf.transpose(-dist)
                    ))
                    
                    # Binary loss from head
                    binary_pred, gesture_pred = self.embedding_net_(sup_x, training=True)
                    binary_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(sup_bin, binary_pred))
                    
                    total_loss = binary_loss + gesture_loss
                
                grads = tape.gradient(total_loss, self.embedding_net_.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.embedding_net_.trainable_variables))
                epoch_loss += total_loss.numpy()
            
            epoch_loss /= steps
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, loss: {epoch_loss:.4f}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        # Store prototypes
        all_emb = self.embedding_extractor_(X_target, training=False).numpy()
        self.prototypes_ = np.array([
            np.mean(all_emb[y_gesture_target == c], axis=0) for c in range(len(self.gesture_classes_))
        ])
        
        return self

    def predict(self, X):
        check_is_fitted(self, ["embedding_net_", "embedding_extractor_", "gesture_encoder_", "prototypes_"])
        X_pad, _ = self._pad(X)
        emb = self.embedding_extractor_(X_pad, training=False).numpy()
        
        # Binary prediction
        binary_pred, _ = self.embedding_net_(X_pad, training=False)
        binary_class = (binary_pred.numpy().flatten() > 0.5).astype(int)
        
        # Gesture prediction (only for those predicted as target)
        dist = self._distance(self.prototypes_, emb)
        gesture_idx = np.argmin(dist, axis=0)
        gesture_pred = self.gesture_encoder_.inverse_transform(gesture_idx)
        
        # Final output: "non_target" for binary=0, otherwise gesture name
        result = np.where(binary_class == 0, "non_target", gesture_pred)
        return result

    def predict_proba(self, X):
        check_is_fitted(self, ["embedding_net_"])
        X_pad, _ = self._pad(X)
        binary_pred, gesture_pred = self.embedding_net_(X_pad, training=False)
        return {"binary": binary_pred.numpy(), "gesture": gesture_pred.numpy()}

    def score(self, X, y):
        from sklearn.metrics import f1_score
        
        y_pred = self.predict(X)
        
        if isinstance(y, pd.DataFrame):
            y_true = y.drop_duplicates("sequence_id").set_index("sequence_id")["gesture"]
            y_true = y_true.reindex(X.index.unique()).fillna("non_target")
        else:
            y_true = pd.Series(y)
        
        # Binary F1 (target vs non_target)
        y_true_binary = (y_true != "non_target").astype(int)
        y_pred_binary = (y_pred != "non_target").astype(int)
        binary_f1 = f1_score(y_true_binary, y_pred_binary)
        
        # Macro F1 on gestures (only target sequences)
        target_mask = y_true != "non_target"
        if target_mask.sum() > 0:
            gesture_f1 = f1_score(y_true[target_mask], y_pred[target_mask], average="macro")
        else:
            gesture_f1 = 0.0
        
        # Competition metric = average of binary F1 and gesture macro F1
        competition_score = (binary_f1 + gesture_f1) / 2
        
        return competition_score