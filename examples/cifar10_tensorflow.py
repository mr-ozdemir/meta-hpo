"""Tune a small convolutional network on CIFAR-10 with TensorFlow and Keras.

Each evaluation trains for three epochs on the first 45,000 training images and scores accuracy on the
last 5,000, so the test set stays untouched for the final model. Needs `pip install meta-hpo tensorflow`.
"""

import tensorflow as tf

import metahpo as mh

(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_fit, y_fit = x_train[:45000], y_train[:45000]
x_val, y_val = x_train[45000:], y_train[45000:]

space = mh.Space(
    {
        "lr": mh.Float(1e-4, 1e-2, log=True),
        "batch_size": mh.Categorical([64, 128, 256]),
        "filters": mh.Int(16, 64),
        "dropout": mh.Float(0.0, 0.5),
        "optimizer": mh.Categorical(["adam", "sgd"]),
    }
)


def build_model(filters, dropout):
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(2 * filters, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(10),
        ]
    )


def train(params):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(0)
    model = build_model(params["filters"], params["dropout"])
    if params["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=params["lr"])
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=params["lr"], momentum=0.9)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(x_fit, y_fit, batch_size=params["batch_size"], epochs=3, verbose=0)
    _, accuracy = model.evaluate(x_val, y_val, verbose=0)
    return accuracy


def main():
    study = mh.optimize(train, space, direction="max", max_evals=30, seed=1)
    print(study.best_params, study.best_score)


if __name__ == "__main__":
    main()
