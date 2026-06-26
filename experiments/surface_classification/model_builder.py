import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras import regularizers

def create_cnn_enc_dec_model(num_shapes, l2_reg):
    model = models.Sequential([
        # Encoder
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
        # layers.Dropout(0.2),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
        # layers.Dropout(0.1),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
        # layers.Dropout(0.2),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
        # layers.Dropout(0.1),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bottleneck
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
        
        # 4️⃣ Dense part
        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=l2_reg),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu', kernel_regularizer=l2_reg),
        layers.Dropout(0.2),
        
        # 5️⃣ Output
        layers.Dense(num_shapes, activation='sigmoid')  # sigmoid for multi-label
    ])
    model.compile(
        optimizer="adam",
        loss='binary_crossentropy',
    )
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

def build_transformer_4classifiers(
        img_size=(16,16,1),
        patch_size=4,
        projection_dim=64,
        transformer_layers=4,
        num_classes=12,
        l2_reg=regularizers.l2(1e-4),
        use_pos_encoding=True):

    inputs = layers.Input(shape=img_size)

    # CNN encoder
    # Encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    # x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    # x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)

    # Patch extraction
    patches = PatchExtract(patch_size)(x)
    num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

    # Patch encoding
    encoded_patches = PatchEncoder(num_patches, projection_dim, use_pos_encoding)(patches)

    # Transformer blocks
    for _ in range(transformer_layers):
        enencoded_patchescoded = transformer_block(encoded_patches, num_heads=4, mlp_dim=128)

    # # Shared features
    features = layers.GlobalAveragePooling1D()(encoded_patches)
    # Classification head
    x = layers.Dense(512)(features)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(0.3)(x)
    res1 = x  # Save for residual
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Add()([x, res1])

    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(0.25)(x)
    res2 = x
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Add()([x, res2])

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(0.2)(x)

    # Four independent classifier heads
    out1 = layers.Dense(num_classes, activation="softmax", name="obj1")(x)
    out2 = layers.Dense(num_classes, activation="softmax", name="obj2")(x)
    out3 = layers.Dense(num_classes, activation="softmax", name="obj3")(x)
    out4 = layers.Dense(num_classes, activation="softmax", name="obj4")(x)

    model = Model(inputs=inputs, outputs=[out1, out2, out3, out4])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss={
        "obj1": "categorical_crossentropy",
        "obj2": "categorical_crossentropy",
        "obj3": "categorical_crossentropy",
        "obj4": "categorical_crossentropy"
        },
        metrics={
        "obj1": "accuracy",
        "obj2": "accuracy",
        "obj3": "accuracy",
        "obj4": "accuracy",
        }
    )
    return model

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

def ViT_on_SE_for_class(surface_model, NUM_CLASSES=12, trainable_body=False, cold_start=False):
    feature_extractor = Model(
        inputs=surface_model.input,
        outputs=surface_model.get_layer(index=-3).output  # the Dense flatten layer before the final Reshape
    )
    feature_extractor.trainable = trainable_body
    x = feature_extractor.output

    x = layers.Dense(512, activation="relu", name="new_dense_1")(x)
    x = layers.Dense(256, activation="relu", name="new_dense_2")(x)

    # Add 4 separate softmax heads
    obj1 = layers.Dense(NUM_CLASSES, activation="softmax", name="obj1")(x)
    obj2 = layers.Dense(NUM_CLASSES, activation="softmax", name="obj2")(x)
    obj3 = layers.Dense(NUM_CLASSES, activation="softmax", name="obj3")(x)
    obj4 = layers.Dense(NUM_CLASSES, activation="softmax", name="obj4")(x)

    # New 4-headed classification model
    classifier_model = Model(inputs=feature_extractor.input, outputs=[obj1, obj2, obj3, obj4])

    classifier_model.compile(
        optimizer="adam",
        loss={
            "obj1": "categorical_crossentropy",
            "obj2": "categorical_crossentropy",
            "obj3": "categorical_crossentropy",
            "obj4": "categorical_crossentropy",
        },
        metrics={
            "obj1": "accuracy",
            "obj2": "accuracy",
            "obj3": "accuracy",
            "obj4": "accuracy",
        }
    )
    if cold_start:
        reset_model_weights(classifier_model)
        print("Cold start enabled — all weights randomly reinitialized.")
    return classifier_model

class MeanValAccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Récupérer les accuracies de validation
        val_accs = [
            logs.get("val_obj1_accuracy"),
            logs.get("val_obj2_accuracy"),
            logs.get("val_obj3_accuracy"),
            logs.get("val_obj4_accuracy")
        ]
        # filtrer les None
        val_accs = [a for a in val_accs if a is not None]
        if val_accs:
            mean_val_acc = sum(val_accs) / len(val_accs)
            print(f"Epoch {epoch+1}: val_mean_accuracy = {mean_val_acc:.4f} | val_loss = {logs.get('val_loss'):.4f}")
