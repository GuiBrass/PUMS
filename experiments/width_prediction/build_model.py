from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, Model
import tensorflow as tf
from tensorflow.keras import regularizers

def build_CNN_model(input_shape, l2_reg):
    model = Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), padding='same'), 
    layers.BatchNormalization(),
    layers.Activation('tanh'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, (3, 3), padding='same'), 
    layers.BatchNormalization(),
    layers.Activation('tanh'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(), 
    layers.Dense(256, activation='tanh', kernel_regularizer=l2_reg),
    layers.Dropout(0.5),
    layers.Dense(128, activation='tanh', kernel_regularizer=l2_reg),
    layers.Dropout(0.5),
    layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer="adam", loss='mae', metrics=['mse'])
    return model

def build_CNN_encoder_decoder(input_shape, l2_reg):
    model = Sequential([

        layers.Input(shape=input_shape),

        # Encoder
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bottleneck
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
        layers.Flatten(), 

        # Decoder
        layers.Dense(512, activation='tanh', kernel_regularizer=l2_reg),
        layers.Dropout(0.3),
        layers.Dense(256, activation='tanh', kernel_regularizer=l2_reg),
        layers.Dropout(0.3),
        layers.Dense(128, activation='tanh', kernel_regularizer=l2_reg),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer="adam", loss='mae', metrics=['mse'])
    return model

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
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(projection_dim)
        self.pos_encoding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.pos_encoding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
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

def build_hybrid_transformer(img_size=(16, 16, 1), patch_size=4, projection_dim=64, transformer_layers=4):
    inputs = layers.Input(shape=img_size)

    # CNN encoder (extract local features)
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)

    # Patch extraction
    patches = PatchExtract(patch_size)(x)
    num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

    # Patch encoding
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Transformer encoder-decoder layers
    for _ in range(transformer_layers):
        encoded_patches = transformer_block(encoded_patches, num_heads=4, mlp_dim=128)

    # Flatten sequence and decode to image
    num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
    x = layers.Reshape((num_patches * projection_dim,))(encoded_patches)
    x = layers.Dense(16 * 62, activation="linear")(x)
    outputs = layers.Reshape((62, 16, 1))(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mae", metrics=["mae"])
    return model

# --- Model definition ---
def build_hybrid_transformer_for_width(img_size=(16, 32, 1), patch_size=4, projection_dim=128, transformer_layers=4, CNN_layers_size=(64,128)):
    l2_reg = regularizers.l2(1e-5)
    inputs = layers.Input(shape=img_size)

    # CNN encoder (local feature extraction)
    # x = layers.Conv2D(CNN_layers_size[0], (3,3), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2_reg)(inputs)
    # x = layers.Conv2D(CNN_layers_size[1], (3,3), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = layers.Conv2D(CNN_layers_size[0], (3,3), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2_reg)(inputs)
    x = layers.Conv2D(CNN_layers_size[1], (3,3), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    
    # Patch extraction
    patches = PatchExtract(patch_size)(x)
    num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

    # Patch encoding
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Transformer encoder layers
    for _ in range(transformer_layers):
        encoded_patches = transformer_block(encoded_patches, num_heads=8, mlp_dim=256)

    # Flatten sequence and decode to scalar output
    x = layers.Flatten()(encoded_patches)
    x = layers.Dense(256, activation="relu")(x)
    # x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(1, activation="linear")(x)  # Single scalar output

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss="mae", metrics=["mae"])
    # model.compile(optimizer="adam", loss="mae", metrics=["mae"])
    # model.summary()
    return model

@tf.keras.utils.register_keras_serializable()
class FeatureExtractor(tf.keras.Model):
    def __init__(self, output_units=62*16, unlocked_weights=False, **kwargs):
        super().__init__(**kwargs)
        self.output_units = output_units
        # On ne stocke pas le modèle complet ici
        self.model_built = False
        self.unlocked_weights = unlocked_weights

    def build(self, input_shape):
        if not self.model_built:
            # Reconstruire ici le "surface_model"
            # Exemple si tu utilises build_hybrid_transformer
            surface_model = build_hybrid_transformer(img_size=input_shape[1:])
            # Prendre la dernière Dense avant reshape
            feature_layer = surface_model.layers[-2].output
            self.model = tf.keras.Model(surface_model.input, feature_layer)
            if self.unlocked_weights == False:
                for layer in self.model.layers:
                    layer.trainable = False
            self.model_built = True
        super().build(input_shape)

    def call(self, x):
        return self.model(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_units": self.output_units
        })
        return config

@tf.keras.utils.register_keras_serializable()
def split_left(x):
    return x[:, :, :16, :]

@tf.keras.utils.register_keras_serializable()
def split_right(x):
    return x[:, :, 16:, :]

def reset_model_weights(model):
    """
    Reset all weights in a keras model by re-running the initializer.
    """
    for layer in model.layers:
        for attr in ["kernel_initializer", "bias_initializer", "recurrent_initializer"]:
            if hasattr(layer, attr.replace("_initializer", "")) and hasattr(layer, attr):
                var = getattr(layer, attr.replace("_initializer", ""))
                initializer = getattr(layer, attr)
                if var is not None and initializer is not None:
                    var.assign(initializer(var.shape, var.dtype))

def build_dual_finger_estimator(surface_model, unlocked_weights=False, cold_start=False):
    # Load extractor twice
    extractor_left  = FeatureExtractor(surface_model, name="feature_extractor_left", unlocked_weights=unlocked_weights)
    extractor_right = FeatureExtractor(surface_model, name="feature_extractor_right", unlocked_weights=unlocked_weights)

    # 16x32 input → split into two 16x16 images
    input_full = layers.Input(shape=(16,32,1))

    x_left  = layers.Lambda(split_left, name="split_left")(input_full)
    x_right = layers.Lambda(split_right, name="split_right")(input_full)

    # Pass through feature extractors
    left_feat  = extractor_left(x_left)
    right_feat = extractor_right(x_right)

    # Concatenate features
    combined = layers.Concatenate()([left_feat, right_feat])

    # Estimation head

    x = layers.Dense(1024)(combined)
    x = layers.Dense(512)(x)
    x = layers.Dense(256)(x)
    x = layers.Dense(128)(x)
    x = layers.Dense(64)(x)
    x = layers.Dense(32)(x)
    output = layers.Dense(1, activation="linear")(x)  # Single scalar output

    model = Model(inputs=input_full, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mae", metrics=["mae"])
    model.summary()

    if cold_start:
        reset_model_weights(model)
        print("⚠️ Cold start enabled — all weights randomly reinitialized.")

    return model