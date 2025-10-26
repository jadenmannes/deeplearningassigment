from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np, pandas as pd, tensorflow as tf, json
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from itertools import product
import math
from copy import deepcopy
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter


# Select if you want to tune the hyperparamaters
tuning = False
TUNING_MODE = "nested"   # options: "nested" or "hpo"

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


# Hyperparamater tuning

def make_optimizer(hp: Dict):
    if hp["opt"] == "sgd":
        return keras.optimizers.SGD(learning_rate=hp["lr"], momentum=hp["momentum"], nesterov=True)
    elif hp["opt"] == "adamw":
        return keras.optimizers.AdamW(learning_rate=hp["lr"], weight_decay=hp["l2"])
    else:
        return keras.optimizers.Adam(learning_rate=hp["lr"])

def split_and_transform(pre: ColumnTransformer, X_df: pd.DataFrame, y_df: pd.DataFrame, labels: np.ndarray):
    Xt, Xtest, yt, ytest, lt, ltest = train_test_split(
        X_df, y_df, labels, test_size=CFG.test_size, random_state=CFG.seed, stratify=labels
    )
    Xtr, Xval, ytr, yval, ltr, lval = train_test_split(
        Xt, yt, lt, test_size=CFG.val_size, random_state=CFG.seed, stratify=lt
    )
    pre_fit = ColumnTransformer(pre.transformers, remainder=pre.remainder) if hasattr(pre, "transformers") else pre
    pre_fit = deepcopy(pre)
    pre_fit.fit(Xtr)
    Xtr_t = pre_fit.transform(Xtr)
    Xval_t = pre_fit.transform(Xval)
    Xtest_t = pre_fit.transform(Xtest)
    ytr_a = ytr.values.astype(np.float32)
    yval_a = yval.values.astype(np.float32)
    ytest_a = ytest.values.astype(np.float32)
    return pre_fit, Xtr_t, ytr_a, ltr, Xval_t, yval_a, lval, Xtest_t, ytest_a, ltest


def cv_score_hp(pre: ColumnTransformer, X_df: pd.DataFrame, y_df: pd.DataFrame, labels: np.ndarray, hp: Dict, folds: int = 5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=CFG.seed)
    f1s, accs = [], []
    for tr_idx, val_idx in skf.split(X_df, labels):
        Xtr_df, Xval_df = X_df.iloc[tr_idx], X_df.iloc[val_idx]
        ytr_df, yval_df = y_df.iloc[tr_idx], y_df.iloc[val_idx]
        ltr, lval = labels[tr_idx], labels[val_idx]
        pre_local = deepcopy(pre)
        pre_local.fit(Xtr_df)
        Xtr_t = pre_local.transform(Xtr_df)
        Xval_t = pre_local.transform(Xval_df)
        ytr = ytr_df.values.astype(np.float32)
        yval = yval_df.values.astype(np.float32)
        model = build_model(Xtr_t.shape[1], y_df.shape[1],
                            width=hp["width"], n_hidden=hp["n_hidden"],
                            l2=hp["l2"], drop=hp["dropout"])
        opt = make_optimizer(hp)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        cw = compute_class_weight("balanced", classes=np.unique(ltr), y=ltr)
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=CFG.patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=max(1, CFG.patience // 4), factor=0.5)
        ]
        model.fit(Xtr_t, ytr, validation_data=(Xval_t, yval),
                  epochs=CFG.epochs, batch_size=hp["batch_size"], verbose=0,
                  callbacks=callbacks, class_weight={i: w for i, w in enumerate(cw)})
        yval_pred = model.predict(Xval_t, verbose=0).argmax(1)
        f1s.append(f1_score(lval, yval_pred, average="macro"))
        accs.append(accuracy_score(lval, yval_pred))
    return float(np.mean(f1s)), float(np.mean(accs))

def search_best_params(pre: ColumnTransformer, X_df: pd.DataFrame, y_df: pd.DataFrame, labels: np.ndarray, max_trials: int = 24, folds: int = 5):
    pre_fit, Xtr_t, ytr, ltr, Xval_t, yval, lval, Xtest_t, ytest, ltest = split_and_transform(pre, X_df, y_df, labels)
    n_classes = y_df.shape[1]
    grid = {
        "opt": ["sgd", "adam", "adamw"],
        "lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "momentum": [0.0, 0.9],
        "n_hidden": [1, 2],
        "width": [16, 32, 64],
        "l2": [0.0, 1e-5, 1e-4, 1e-3],
        "dropout": [0.0, 0.2, 0.5],
        "batch_size": [16, 32]
    }

    keys = list(grid.keys())
    all_trials = list(product(*[grid[k] for k in keys]))
    baseline = {"opt": "sgd", "lr": 1e-3, "momentum": 0.9, "n_hidden": 2, "width": 32, "l2": 1e-4, "dropout": 0.2, "batch_size": 16}
    def hp_equal(a, b): return all(a[k] == b[k] for k in a)
    if not any(hp_equal(baseline, {k: v for k, v in zip(keys, combo)}) for combo in all_trials):
        all_trials = [tuple(baseline[k] for k in keys)] + all_trials
    if len(all_trials) > max_trials:
        rng = np.random.default_rng(CFG.seed)
        choices = rng.choice(len(all_trials), size=max_trials, replace=False)
        all_trials = [all_trials[i] for i in choices]
        if not any(hp_equal(baseline, {k: v for k, v in zip(keys, combo)}) for combo in all_trials):
            all_trials[0] = tuple(baseline[k] for k in keys)
    results = []
    best = {"score": -1.0, "acc": -1.0, "hp": None}
    for combo in all_trials:
        hp = {k: v for k, v in zip(keys, combo)}
        if hp["opt"] != "sgd":
            hp["momentum"] = 0.0
        mean_f1, mean_acc = cv_score_hp(pre, X_df, y_df, labels, hp, folds=folds)
        results.append({"hp": hp, "cv_f1m": mean_f1, "cv_acc": mean_acc})
        better = (mean_f1 > best["score"]) or (math.isclose(mean_f1, best["score"], rel_tol=1e-4) and mean_acc > best["acc"]) or (math.isclose(mean_f1, best["score"], rel_tol=1e-4) and math.isclose(mean_acc, best["acc"], rel_tol=1e-4) and (hp["n_hidden"] * hp["width"] < (best["hp"]["n_hidden"] * best["hp"]["width"]) if best["hp"] else True))
        if better:
            best["score"] = mean_f1
            best["acc"] = mean_acc
            best["hp"] = hp

    pre_refit = ColumnTransformer(pre.transformers, remainder=pre.remainder) if hasattr(pre, "transformers") else pre
    pre_refit = deepcopy(pre)

    Xt, Xtest, yt, ytest, lt, ltest = train_test_split(X_df, y_df, labels, test_size=CFG.test_size, random_state=CFG.seed, stratify=labels)
    Xtv, Xval_hold, ytv, yval_hold, ltv, lval_hold = train_test_split(Xt, yt, lt, test_size=CFG.val_size, random_state=CFG.seed, stratify=lt)

    pre_refit.fit(Xtv)
    Xtv_t = pre_refit.transform(Xtv)
    Xval_hold_t = pre_refit.transform(Xval_hold)
    Xtest_t = pre_refit.transform(Xtest)
    ytv_a = ytv.values.astype(np.float32)
    yval_hold_a = yval_hold.values.astype(np.float32)

    best_hp = best["hp"]
    model_star = build_model(Xtv_t.shape[1], n_classes, width=best_hp["width"], n_hidden=best_hp["n_hidden"], l2=best_hp["l2"], drop=best_hp["dropout"])
    opt_star = make_optimizer(best_hp)
    model_star.compile(loss="categorical_crossentropy", optimizer=opt_star, metrics=["accuracy"])
    cw_tv = compute_class_weight("balanced", classes=np.unique(ltv), y=ltv)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=CFG.patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=max(1, CFG.patience // 4), factor=0.5)
    ]
    model_star.fit(Xtv_t, ytv_a, validation_data=(Xval_hold_t, yval_hold_a), epochs=CFG.epochs, batch_size=best_hp["batch_size"], verbose=0, callbacks=callbacks, class_weight={i: w for i, w in enumerate(cw_tv)})
    ytest_pred = model_star.predict(Xtest_t, verbose=0).argmax(1)
    test_acc = accuracy_score(ltest, ytest_pred)
    test_f1m = f1_score(ltest, ytest_pred, average="macro")
    return {"best_hp": best_hp, "cv_results": results, "model": model_star, "pre": pre_refit, "test_acc": float(test_acc), "test_f1": float(test_f1m)}

# Nested CV

def inner_search_best_hp(pre, X_df, y_df, labels, grid, folds=3):
    keys = list(grid.keys())
    all_trials = list(product(*[grid[k] for k in keys]))
    baseline = {"opt": "sgd", "lr": 1e-3, "momentum": 0.9, "n_hidden": 2, "width": 32, "l2": 1e-4, "dropout": 0.2, "batch_size": 16}
    def hp_equal(a, b): return all(a[k] == b[k] for k in a)
    if not any(hp_equal(baseline, {k: v for k, v in zip(keys, combo)}) for combo in all_trials):
        all_trials = [tuple(baseline[k] for k in keys)] + all_trials
    rng = np.random.default_rng(CFG.seed)
    if len(all_trials) > 24:
        choices = rng.choice(len(all_trials), size=24, replace=False)
        all_trials = [all_trials[i] for i in choices]
        if not any(hp_equal(baseline, {k: v for k, v in zip(keys, combo)}) for combo in all_trials):
            all_trials[0] = tuple(baseline[k] for k in keys)
    best = {"score": -1.0, "acc": -1.0, "hp": None}
    for combo in all_trials:
        hp = {k: v for k, v in zip(keys, combo)}
        if hp["opt"] != "sgd":
            hp["momentum"] = 0.0
        f1m, acc = cv_score_hp(pre, X_df, y_df, labels, hp, folds=folds)
        better = (f1m > best["score"]) or (math.isclose(f1m, best["score"], rel_tol=1e-4) and acc > best["acc"]) or (math.isclose(f1m, best["score"], rel_tol=1e-4) and math.isclose(acc, best["acc"], rel_tol=1e-4) and (hp["n_hidden"] * hp["width"] < (best["hp"]["n_hidden"] * best["hp"]["width"]) if best["hp"] else True))
        if better:
            best["score"], best["acc"], best["hp"] = float(f1m), float(acc), hp
    return best["hp"]

def fit_on_outer_train(pre, X_df_tr, y_df_tr, labels_tr, hp):
    pre_fit = deepcopy(pre)
    Xtr_df, Xval_df, ytr_df, yval_df, ltr, lval = train_test_split(X_df_tr, y_df_tr, labels_tr, test_size=CFG.val_size, random_state=CFG.seed, stratify=labels_tr)
    pre_fit.fit(Xtr_df)
    Xtr = pre_fit.transform(Xtr_df)
    Xval = pre_fit.transform(Xval_df)
    ytr = ytr_df.values.astype(np.float32)
    yval = yval_df.values.astype(np.float32)
    model = build_model(Xtr.shape[1], y_df_tr.shape[1], width=hp["width"], n_hidden=hp["n_hidden"], l2=hp["l2"], drop=hp["dropout"])
    opt = make_optimizer(hp)
    model.compile(loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05), optimizer=opt, metrics=["accuracy"])
    classes = np.unique(ltr)
    cw = compute_class_weight("balanced", classes=classes, y=ltr)
    class_weight = {int(cls): float(w) for cls, w in zip(classes, cw)}
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=CFG.patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=max(1, CFG.patience // 4), factor=0.5),
        keras.callbacks.ModelCheckpoint("outputs/best_outer.keras", monitor="val_loss", save_best_only=True)
    ]
    model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=CFG.epochs, batch_size=hp["batch_size"], verbose=0, callbacks=callbacks, class_weight=class_weight)
    return model, pre_fit

def nested_cv(df, outer_folds=5, inner_folds=3):
    pre_all, X_all, y_all, class_names = make_preprocessor(df)
    labels_all = y_all.values.argmax(1)
    skf_outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=CFG.seed)
    grid = {
        "opt": ["sgd", "adam", "adamw"],
        "lr": [1e-4, 3e-4, 1e-3, 3e-3],
        "momentum": [0.0, 0.9],
        "n_hidden": [1, 2],
        "width": [16, 32, 64],
        "l2": [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
        "dropout": [0.0, 0.2, 0.5],
        "batch_size": [16, 32]
    }
    fold_results = []
    all_best_hps = []
    for fold_idx, (tr_idx, te_idx) in enumerate(skf_outer.split(X_all, labels_all), 1):
        X_tr_df, X_te_df = X_all.iloc[tr_idx].reset_index(drop=True), X_all.iloc[te_idx].reset_index(drop=True)
        y_tr_df, y_te_df = y_all.iloc[tr_idx].reset_index(drop=True), y_all.iloc[te_idx].reset_index(drop=True)
        labels_tr, labels_te = labels_all[tr_idx], labels_all[te_idx]
        pre_fold = deepcopy(pre_all)
        best_hp = inner_search_best_hp(pre_fold, X_tr_df, y_tr_df, labels_tr, grid, folds=inner_folds)
        all_best_hps.append(best_hp)
        model, pre_fit = fit_on_outer_train(pre_fold, X_tr_df, y_tr_df, labels_tr, best_hp)
        Xte = pre_fit.transform(X_te_df)
        yte = y_te_df.values.astype(np.float32)
        ypred = model.predict(Xte, verbose=0).argmax(1)
        acc = accuracy_score(labels_te, ypred)
        f1m = f1_score(labels_te, ypred, average="macro")
        fold_results.append({"fold": fold_idx, "acc": float(acc), "macro_f1": float(f1m), "hp": best_hp})
    accs = np.array([r["acc"] for r in fold_results], dtype=float)
    f1s = np.array([r["macro_f1"] for r in fold_results], dtype=float)
    summary = {
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std(ddof=1)),
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std(ddof=1)),
        "per_fold": fold_results,
        "best_hps": all_best_hps
    }
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    # load data
    df = load_data("Data/palmerpenguins.csv")

    if tuning and TUNING_MODE == "hpo":
        pre, X_df, y_df, class_names = make_preprocessor(df)
        labels = y_df.values.argmax(1)
        out = search_best_params(pre, X_df, y_df, labels, max_trials=24, folds=5)
        print(json.dumps(out["best_hp"], indent=2))
        print(f"{out['test_acc']:.6f}")
        print(f"{out['test_f1']:.6f}")

    elif tuning and TUNING_MODE == "nested":
        # Run unbiased nested CV estimate
        summary = nested_cv(df, outer_folds=5, inner_folds=3)
        print("\n=== Nested CV summary ===")
        print(f"Outer-CV Accuracy : {summary['acc_mean']:.6f} ± {summary['acc_std']:.6f}")
        print(f"Outer-CV Macro-F1 : {summary['f1_mean']:.6f} ± {summary['f1_std']:.6f}")


        def hp_key(h):  # make dict hashable and consistently ordered
            return tuple((k, h[k]) for k in sorted(h.keys()))


        counts = Counter(hp_key(h) for h in summary["best_hps"])
        consensus_hp = dict(counts.most_common(1)[0][0])

        print("\nConsensus HP from outer folds:")
        print(json.dumps(consensus_hp, indent=2))

        pre, X_df, y_df, class_names = make_preprocessor(df)
        labels = y_df.values.argmax(1)
        model, pre_fitted, acc, f1m, hist, y_true, y_pred = train_once(
            pre, X_df, y_df, labels, consensus_hp, return_history=True
        )
        print("\nFinal hold-out (single split) for deployable model:")
        print(f"Test Accuracy: {acc:.6f}")
        print(f"Macro-F1    : {f1m:.6f}")

    else:
        # pre-porcess
        pre, X, y, class_names = make_preprocessor(df)
        labels = y.values.argmax(1)

        # Define hyperparameters (take from tuned nested cv)
        hp = {
            "opt": "sgd",
            "lr": 0.0003,
            "momentum": 0.9,
            "n_hidden": 1,
            "width": 16,
            "l2": 1e-05,
            "dropout": 0.0,
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

