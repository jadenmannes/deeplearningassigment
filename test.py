from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

@dataclass
class CFG:
    seed: int = 42
    test_size: float = 0.20
    val_size: float = 0.25
    epochs: int = 200
    patience: int = 20
CFG = CFG()

tf.keras.utils.set_random_seed(CFG.seed)

try:
    tf.config.experimental.enable_op_determinism()
except TypeError:
    tf.config.experimental.enable_op_determinism(True)

def load_data(path:str)->pd.DataFrame:
    df = pd.read_csv(path).dropna()
    return df

def make_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, pd.DataFrame, pd.DataFrame, List[str]]:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols + ['species']]
    y_df = pd.get_dummies(df['species'])
    X_df = df.drop(columns=['species'])


    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe,             cat_cols)
        ]
    )
    return pre, X_df, y_df, y_df.columns.to_list()

def build_model(n_in:int, n_classes:int, width:int=32, n_hidden:int=2, l2=1e-4, drop=0.2):
    reg = keras.regularizers.l2(l2) if l2>0 else None
    x = keras.Input(shape=(n_in,))
    h = x
    for _ in range(n_hidden):
        h = layers.Dense(width, activation="relu", kernel_regularizer=reg)(h)
        h = layers.BatchNormalization()(h)
        if drop>0: h = layers.Dropout(drop)(h)
    y = layers.Dense(n_classes, activation="softmax")(h)
    return keras.Model(x, y)

def train_once(pre:ColumnTransformer, X_df:pd.DataFrame, y_df:pd.DataFrame, labels:np.ndarray, hp:Dict, return_history:bool=True):
    Xt, Xtest, yt, ytest, lt, ltest = train_test_split(
        X_df, y_df, labels, test_size=CFG.test_size, random_state=CFG.seed, stratify=labels
    )
    Xtr, Xval, ytr, yval, ltr, lval = train_test_split(
        Xt, yt, lt, test_size=CFG.val_size, random_state=CFG.seed, stratify=lt
    )

    pre.fit(Xtr)
    Xtr   = pre.transform(Xtr)
    Xval  = pre.transform(Xval)
    Xtest = pre.transform(Xtest)

    ytr   = ytr.values.astype(np.float32)
    yval  = yval.values.astype(np.float32)
    ytest = ytest.values.astype(np.float32)

    model = build_model(Xtr.shape[1], y_df.shape[1],
                        width=hp["width"], n_hidden=hp["n_hidden"],
                        l2=hp["l2"], drop=hp["dropout"])
    if hp["opt"]=="sgd":
        opt = keras.optimizers.SGD(learning_rate=hp["lr"], momentum=hp["momentum"], nesterov=True)
    elif hp["opt"]=="adamw":
        opt = keras.optimizers.AdamW(learning_rate=hp["lr"], weight_decay=hp["l2"])
    else:
        opt = keras.optimizers.Adam(learning_rate=hp["lr"])

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    cw = compute_class_weight("balanced", classes=np.unique(ltr), y=ltr)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=CFG.patience, restore_best_weights=True, monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(patience=max(1, CFG.patience // 4), factor=0.5)
    ]

    history = model.fit(Xtr, ytr, validation_data=(Xval, yval),
                        epochs=CFG.epochs, batch_size=hp["batch_size"], verbose=0,
                        callbacks=callbacks, class_weight={i: w for i, w in enumerate(cw)})

    # evaluate
    yprob = model.predict(Xtest, verbose=0)
    ypred = yprob.argmax(1)
    acc = accuracy_score(ltest, ypred)
    f1m = f1_score(ltest, ypred, average="macro")

    if return_history:
        return model, pre, acc, f1m, history.history, ltest, ypred
    else:
        return model, pre, acc, f1m



if __name__ == "__main__":
    # load data
    df = load_data("Data/palmerpenguins.csv")

    # pre-porcess
    pre, X, y, class_names = make_preprocessor(df)
    labels = y.values.argmax(1)

    # Define hyperparameters
    hp = {
        "opt": "sgd",
        "lr": 0.001,
        "momentum": 0.9,
        "n_hidden": 2,
        "width": 32,
        "l2": 1e-4,
        "dropout": 0.2,
        "batch_size": 16
    }

    # Train once
    model, pre_fitted, acc, f1m, hist, y_true, y_pred = train_once(pre, X, y, labels, hp, return_history=True)

    # Results
    print(f"Test Accuracy: {acc:.3f}")
    print(f"Macro-F1: {f1m:.3f}")

    # Training curves: Accuracy
    plt.figure(figsize=(7, 4))
    plt.plot(hist["accuracy"], label="train acc")
    plt.plot(hist["val_accuracy"], label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Training curves: Loss
    plt.figure(figsize=(7, 4))
    plt.plot(hist["loss"], label="train loss")
    plt.plot(hist["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Confusion Matrix (normalized)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true), normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, values_format=".2f", cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.show()

    # Save model + preprocessor ===
    # model.save("models/penguin_model.keras")
    # import joblib
    # joblib.dump(pre_fitted, "models/preprocessor.pkl")