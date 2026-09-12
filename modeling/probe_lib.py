import numpy as np

D = {"general": 768, "satellite": 1024}

def load_block(split_name, column, key, pooling="cls"):
    F = np.load(f"feats/{split_name}.{column}.{key}.npy")     # columns [:D] are CLS, [D:] are the patch mean
    d = D[key]
    return {"cls": F[:, :d], "patch": F[:, d:], "both": F}[pooling]

BLOCKS = {"ground": [("ground_image", "general")],
          "drone":  [("drone_image",  "satellite")],
          "concat": [("ground_image", "general"), ("drone_image", "satellite")]}   # 768 + 1024 = 1,792

def features(split_name, modality, rows=None):
    X = np.concatenate([load_block(split_name, c, k) for c, k in BLOCKS[modality]], axis=1)
    return X if rows is None else X[rows]                      # rows = pool.index for the CV pool

def standardize(X_tr, X_va):
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6                  # training-fold statistics only
    return (X_tr - mu) / sd, (X_va - mu) / sd


if __name__ == "__main__":
    for mod in BLOCKS:
        X = features("train", mod)
        print(f"{mod:7s} {X.shape}  dtype {X.dtype}")
    # standardize smoke test on an arbitrary split
    X = features("train", "concat")
    tr, va = standardize(X[:5000], X[5000:])
    print(f"standardized: train mean {tr.mean():+.3e} std {tr.std():.3f}   "
          f"val mean {va.mean():+.3e} std {va.std():.3f}")
    for pool in ("cls", "patch", "both"):
        b = load_block("train", "ground_image", "general", pool)
        print(f"load_block ground/general {pool:5s} -> {b.shape}")
