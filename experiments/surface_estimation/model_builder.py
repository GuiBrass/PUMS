import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K

def create_patches(x, patch_size=4):
    # Extracts non-overlapping patches of size 4x4
    patches = tf.image.extract_patches(
        images=x,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    # Flatten the patches
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [tf.shape(patches)[0], -1, patch_dims])
    return patches

@tf.keras.utils.register_keras_serializable()
class PatchExtract(layers.Layer):
    def __init__(self, patch_size=4, **kwargs):
        super().__init__(**kwargs)  # pass name, dtype, trainable, etc.
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [tf.shape(patches)[0], -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, use_pos_encoding, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.use_pos_encoding = use_pos_encoding
        self.projection_dim = projection_dim
        self.projection = layers.Dense(projection_dim)
        self.pos_encoding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        if self.use_pos_encoding:
            encoded = self.projection(patches) + self.pos_encoding(positions)
        else:
            encoded = self.projection(patches)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
            "use_pos_encoding": self.use_pos_encoding,
        })
        return config

# --- Transformer block ---
def transformer_block(x, num_heads, mlp_dim, dropout_rate=0.1):
    # Multi-head self-attention
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    # MLP block
    mlp_output = layers.Dense(mlp_dim, activation='gelu')(x)
    mlp_output = layers.Dense(x.shape[-1])(mlp_output)
    mlp_output = layers.Dropout(dropout_rate)(mlp_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + mlp_output)
    return x

# --- Model definition ---
def build_hybrid_transformer(
    img_size=(16,16,1),
    cnn_size=[32, 64],
    patch_size=4,
    projection_dim=64,
    transformer_layers=4,
    use_cnn=True,
    use_transformer=True,
    use_pos_encoding=True
):
    
    inputs = layers.Input(shape=img_size)
    x = inputs

    # CNN encoder (extract local features)
    if use_cnn: 
        x = layers.Conv2D(cnn_size[0], (3,3), padding="same", activation="relu")(x)
        x = layers.Conv2D(cnn_size[1], (3,3), padding="same", activation="relu")(x)

    # Patch extraction
    patches = PatchExtract(patch_size)(x)
    num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

    # Patch encoding
    encoded_patches = PatchEncoder(num_patches, projection_dim, use_pos_encoding)(patches)

    # Transformer encoder-decoder layers
    if use_transformer:
        for _ in range(transformer_layers):
            encoded_patches = transformer_block(encoded_patches, num_heads=4, mlp_dim=128)

    # Flatten sequence and decode to image
    num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
    x = layers.Reshape((num_patches * projection_dim,))(encoded_patches)
    x = layers.Dense(15 * 60, activation="linear")(x)
    outputs = layers.Reshape((60, 15, 1))(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mae", metrics=["mae"])
    return model


@tf.keras.utils.register_keras_serializable()
def rmse_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

@tf.keras.utils.register_keras_serializable()
def rmse_loss_threshold(y_true, y_pred):
    threshold = 0.22  # threshold in normalized units

    error = K.abs(y_true - y_pred)

    # Soft threshold: remove small errors
    error = K.maximum(error - threshold, 0.0)

    return K.sqrt(K.mean(K.square(error)))