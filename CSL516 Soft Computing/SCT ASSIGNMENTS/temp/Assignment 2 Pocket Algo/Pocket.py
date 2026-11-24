# Pocket Algorithm on the Lung Cancer survey dataset (YES = 1, NO = 0)
# Pure Python (no numpy / pandas); matplotlib is used ONLY to draw the graphs.
# No keyboard input: everything is read from / written to files.
# Weights and bias start as random numbers in [-0.5, 0.5] (own LCG generator).
#
# Files (same folder as this script):
#   survey_lung_cancer.csv -> original dataset (input)
#   lung_train.csv / lung_test1.csv / lung_test2.csv -> created automatically (80/10/10)

import os
import matplotlib.pyplot as plt

SOURCE_FILE = "survey_lung_cancer.csv"
TRAIN_FILE = "lung_train.csv"
TEST_FILES = ["lung_test1.csv", "lung_test2.csv"]


# ---------------- own shuffle (no random module) ----------------
def shuffled(items, seed=42):
    items = list(items)
    state = seed
    for i in range(len(items) - 1, 0, -1):
        state = (1103515245 * state + 12345) % (2 ** 31)
        j = state % (i + 1)
        items[i], items[j] = items[j], items[i]
    return items


# ---------------- small random weight generator (no random module) ----------------
def init_params(m, seed=2024):
    """Returns m weights and 1 bias, each a random number in [-0.5, 0.5].
    Uses a simple linear congruential generator (LCG), so results are repeatable."""
    state = seed
    vals = []
    for _ in range(m + 1):
        state = (1103515245 * state + 12345) % (2 ** 31)
        vals.append(state / float(2 ** 31) - 0.5)
    return vals[:m], vals[m]


# ---------------- MODULE 0: split source into train / test1 / test2 ----------------
def prepare_files():
    if all(os.path.exists(f) for f in [TRAIN_FILE] + TEST_FILES):
        return
    if not os.path.exists(SOURCE_FILE):
        print("[Error] File not found:", SOURCE_FILE)
        return
    with open(SOURCE_FILE) as f:
        lines = [l.strip() for l in f if l.strip()]
    header = [h.strip() for h in lines[0].split(",")]
    feature_names = header[:-1]                  # last column = LUNG_CANCER

    seen, no_cls, yes_cls = set(), [], []
    for line in lines[1:]:
        if line in seen:                         # remove exact duplicate rows
            continue
        seen.add(line)
        t = [v.strip() for v in line.split(",")]
        feats = [1.0 if t[0] == "M" else 0.0]    # GENDER: M = 1, F = 0
        feats.append(float(t[1]))                # AGE
        feats += [float(v) - 1.0 for v in t[2:-1]]   # symptoms: 1/2 -> 0/1
        (yes_cls if t[-1] == "YES" else no_cls).append(feats + [1 if t[-1] == "YES" else 0])

    train, test1, test2 = [], [], []
    for group in (no_cls, yes_cls):              # stratified 80/10/10 per class
        group = shuffled(group)
        n = len(group)
        n_train = int(0.8 * n)
        n_t1 = (n - n_train) // 2
        train += group[:n_train]
        test1 += group[n_train:n_train + n_t1]
        test2 += group[n_train + n_t1:]

    for fname, rows in zip([TRAIN_FILE] + TEST_FILES, [shuffled(train, 7), test1, test2]):
        with open(fname, "w") as out:
            out.write(",".join(feature_names) + ",class\n")
            for r in rows:
                out.write(",".join(str(v) for v in r[:-1]) + "," + str(r[-1]) + "\n")
    print(f"[Data] {len(no_cls) + len(yes_cls)} unique rows -> {TRAIN_FILE} ({len(train)}), "
          f"{TEST_FILES[0]} ({len(test1)}), {TEST_FILES[1]} ({len(test2)})")


# ---------------- utilities ----------------
def read_csv(filename):
    X, y, names = [], [], []
    with open(filename) as f:
        for k, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            tok = line.split(",")
            if k == 0:
                names = tok[:-1]
                continue
            X.append([float(v) for v in tok[:-1]])
            y.append(int(float(tok[-1])))
    return X, y, names


def min_max_fit(X):
    m = len(X[0])
    return [min(r[j] for r in X) for j in range(m)], [max(r[j] for r in X) for j in range(m)]


def min_max_apply(X, mins, maxs):
    return [[(r[j] - mins[j]) / (maxs[j] - mins[j]) if maxs[j] != mins[j] else 0.0
             for j in range(len(r))] for r in X]


def dot(a, b):
    s = 0.0
    for p, q in zip(a, b):
        s += p * q
    return s


def step(v):
    return 1 if v >= 0 else 0


def accuracy(X, y, w, b):
    ok = 0
    for i in range(len(X)):
        if step(dot(w, X[i]) + b) == y[i]:
            ok += 1
    return 100.0 * ok / len(X)


def load_all(train_file, test_files):
    """Train data defines the scaling; test data is scaled with the same values."""
    Xtr, ytr, names = read_csv(train_file)
    mins, maxs = min_max_fit(Xtr)
    Xtr = min_max_apply(Xtr, mins, maxs)
    tests = []
    for tf in test_files:
        Xt, yt, _ = read_csv(tf)
        tests.append((tf, min_max_apply(Xt, mins, maxs), yt))
    return Xtr, ytr, names, tests


def confusion(X, y, w, b):
    tp = tn = fp = fn = 0
    for i in range(len(X)):
        p = step(dot(w, X[i]) + b)
        if p == 1 and y[i] == 1: tp += 1
        elif p == 0 and y[i] == 0: tn += 1
        elif p == 1 and y[i] == 0: fp += 1
        else: fn += 1
    return tp, tn, fp, fn


# ---------------- MODULE 2: TESTING ----------------
def test_model(title, tests, w, b):
    print("\n" + "=" * 66)
    print(f" MODULE 2 : TESTING  ({len(tests)} test files, {title})")
    print("=" * 66)
    for k, (fname, X, y) in enumerate(tests, 1):
        tp, tn, fp, fn = confusion(X, y, w, b)
        total = len(X)
        print(f"\nTest file {k}: {fname}  ({total} samples)")
        print(f"  Confusion matrix: TP={tp}  TN={tn}  FP={fp}  FN={fn}")
        print(f"  Accuracy = {tp + tn}/{total} = {100.0 * (tp + tn) / total:.2f}%")


def majority_baseline(y):
    ones = sum(y)
    return 100.0 * max(ones, len(y) - ones) / len(y)

# ---------------- settings ----------------
LEARNING_RATE = 0.1
MAX_EPOCHS = 300


def count_errors(X, y, w, b):
    err = 0
    for i in range(len(X)):
        if step(dot(w, X[i]) + b) != y[i]:
            err += 1
    return err


# ---------------- MODULE 1: TRAINING ----------------
def train_pocket(Xtr, ytr, names, tests):
    print("\n" + "=" * 70)
    print(" MODULE 1 : TRAINING  (Pocket Algorithm)")
    print(f" Train samples: {len(Xtr)} | LR = {LEARNING_RATE} | Max epochs = {MAX_EPOCHS}")
    print(f" Majority-class baseline accuracy (train) = {majority_baseline(ytr):.2f}%")
    print("=" * 70)

    n, m = len(Xtr), len(Xtr[0])
    w, b = init_params(m)                        # random start in [-0.5, 0.5]
    pw, pb = w[:], b
    p_err = count_errors(Xtr, ytr, pw, pb)

    epochs, e_pocket, e_current = [], [], []
    acc_train, acc_t1, acc_t2 = [], [], []
    print(f"{'Epoch':<7}{'Cur Err':<9}{'Pocket Err':<12}{'E_avg(pocket)':<15}{'Train Acc %':<13}")
    print("-" * 56)

    for ep in range(1, MAX_EPOCHS + 1):
        for i in shuffled(range(n), seed=ep):
            e = ytr[i] - step(dot(w, Xtr[i]) + b)
            if e != 0:
                for j in range(m):
                    w[j] += LEARNING_RATE * e * Xtr[i][j]
                b += LEARNING_RATE * e
                cur = count_errors(Xtr, ytr, w, b)
                if cur < p_err:                  # pocket: keep only the best weights
                    pw, pb, p_err = w[:], b, cur
        epochs.append(ep)
        e_current.append(count_errors(Xtr, ytr, w, b) / n)
        e_pocket.append(p_err / n)
        acc_train.append(accuracy(Xtr, ytr, pw, pb))
        acc_t1.append(accuracy(tests[0][1], tests[0][2], pw, pb))
        acc_t2.append(accuracy(tests[1][1], tests[1][2], pw, pb))
        if ep <= 10 or ep % 10 == 0 or p_err == 0:
            print(f"{ep:<7}{int(e_current[-1] * n):<9}{p_err:<12}{e_pocket[-1]:<15.4f}{acc_train[-1]:<13.2f}")
        if p_err == 0:
            print(f"\n>>> Pocket reached 0 errors at epoch {ep} <<<")
            break

    print(f"\nBest (pocket) training errors = {p_err}/{n} -> E_average = {p_err / n:.4f}")
    print("Pocket weights:")
    for j, name in enumerate(names):
        print(f"  w[{name}] = {pw[j]:.4f}")
    print(f"  bias = {pb:.4f}")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 4.8))
    a1.bar(epochs, e_pocket, color="#4C72B0")
    a1.set_title("Graph 1: E_average of Pocket (per epoch)")
    a1.set_xlabel("Epoch"); a1.set_ylabel("E_average"); a1.grid(axis="y", linestyle="--", alpha=0.5)
    a2.plot(epochs, e_current, color="#999999", linewidth=1.2, label="Current perceptron")
    a2.plot(epochs, e_pocket, color="#D62728", linewidth=2.2, label="Pocket (best so far)")
    a2.set_title("Graph 2: E_average vs Epoch")
    a2.set_xlabel("Epoch"); a2.set_ylabel("E_average"); a2.grid(True, linestyle="--", alpha=0.5); a2.legend()
    a3.plot(epochs, acc_train, color="#2CA02C", linewidth=2.2, label="Train accuracy")
    a3.plot(epochs, acc_t1, color="#1F77B4", linewidth=1.5, linestyle="--", label="Test 1 accuracy")
    a3.plot(epochs, acc_t2, color="#FF7F0E", linewidth=1.5, linestyle="--", label="Test 2 accuracy")
    a3.set_title("Graph 3: Accuracy vs Epoch (pocket weights)")
    a3.set_xlabel("Epoch"); a3.set_ylabel("Accuracy (%)"); a3.set_ylim(0, 105)
    a3.grid(True, linestyle="--", alpha=0.5); a3.legend()
    plt.tight_layout()
    plt.savefig("pocket_graphs.png", dpi=150)
    print("\n[Saved] pocket_graphs.png")
    plt.show()
    return pw, pb


if __name__ == "__main__":
    prepare_files()
    Xtr, ytr, names, tests = load_all(TRAIN_FILE, TEST_FILES)
    pw, pb = train_pocket(Xtr, ytr, names, tests)
    test_model("POCKET weights", tests, pw, pb)