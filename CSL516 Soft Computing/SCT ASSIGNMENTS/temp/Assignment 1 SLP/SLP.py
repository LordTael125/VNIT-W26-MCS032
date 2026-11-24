# Single Layer Perceptron on the Iris dataset (Setosa vs Not-Setosa)
# Pure Python (no numpy / pandas). matplotlib is used ONLY to draw the 2 graphs.
# No keyboard input: everything is read from / written to files.
# Weights and bias start as random numbers in [-0.5, 0.5] (own LCG generator).
#
# Files (same folder as this script):
#   iris.data      -> original UCI dataset (input; bezdekIris.data also works if renamed)
#   iris_train.csv -> training set (auto-created, 80%)
#   iris_test1.csv -> test set 1   (auto-created, 10%)
#   iris_test2.csv -> test set 2   (auto-created, 10%)
# Class 0 = Iris-setosa, Class 1 = Iris-versicolor / Iris-virginica

import os
import matplotlib.pyplot as plt

SOURCE_FILE = "iris.data"
TRAIN_FILE = "iris_train.csv"
TEST_FILES = ["iris_test1.csv", "iris_test2.csv"]
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
LEARNING_RATE = 0.1
MAX_EPOCHS = 200


# ---------------- own shuffle (no random module) ----------------
def shuffled(items, seed=42):
    items = list(items)
    state = seed
    for i in range(len(items) - 1, 0, -1):
        state = (1103515245 * state + 12345) % (2 ** 31)
        j = state % (i + 1)
        items[i], items[j] = items[j], items[i]
    return items


# ---------------- MODULE 0: split source into train / test1 / test2 ----------------
def prepare_files():
    if all(os.path.exists(f) for f in [TRAIN_FILE] + TEST_FILES):
        return
    if not os.path.exists(SOURCE_FILE):
        print("[Error] File not found:", SOURCE_FILE)
        return
    setosa, other = [], []
    with open(SOURCE_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = line.split(",")
            feats = [float(v) for v in t[:4]]
            if t[4].strip() == "Iris-setosa":
                setosa.append(feats + [0])
            else:
                other.append(feats + [1])

    train, test1, test2 = [], [], []
    for group in (setosa, other):                # stratified: each class split 80/10/10
        group = shuffled(group)
        n = len(group)
        n_train = int(0.8 * n)
        n_t1 = (n - n_train) // 2
        train += group[:n_train]
        test1 += group[n_train:n_train + n_t1]
        test2 += group[n_train + n_t1:]

    for fname, rows in zip([TRAIN_FILE] + TEST_FILES, [shuffled(train, 7), test1, test2]):
        with open(fname, "w") as out:
            out.write(",".join(FEATURES) + ",class\n")
            for r in rows:
                out.write(",".join(str(v) for v in r[:-1]) + "," + str(r[-1]) + "\n")
    print(f"[Data] Created {TRAIN_FILE} ({len(train)}), {TEST_FILES[0]} ({len(test1)}), "
          f"{TEST_FILES[1]} ({len(test2)})")


# ---------------- utilities ----------------
def read_csv(filename):
    X, y = [], []
    with open(filename) as f:
        for k, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            tok = line.split(",")
            if k == 0:
                try:
                    float(tok[0])
                except ValueError:
                    continue
            X.append([float(v) for v in tok[:-1]])
            y.append(int(float(tok[-1])))
    return X, y


def dot(a, b):
    s = 0.0
    for p, q in zip(a, b):
        s += p * q
    return s


def step(v):
    return 1 if v >= 0 else 0


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


def accuracy(X, y, w, b):
    ok = 0
    for i in range(len(X)):
        if step(dot(w, X[i]) + b) == y[i]:
            ok += 1
    return 100.0 * ok / len(X)


# ---------------- MODULE 1: TRAINING ----------------
def train_slp(train_file, test_files):
    print("\n" + "=" * 60)
    print(" MODULE 1 : TRAINING  (Single Layer Perceptron)")
    print(f" Train file: {train_file} | LR = {LEARNING_RATE} | Max epochs = {MAX_EPOCHS}")
    print("=" * 60)

    X, y = read_csv(train_file)
    t1X, t1y = read_csv(test_files[0])
    t2X, t2y = read_csv(test_files[1])
    n, m = len(X), len(X[0])
    w, b = init_params(m)                        # random start in [-0.5, 0.5]
    epochs, e_avg_list = [], []
    acc_train, acc_t1, acc_t2 = [], [], []

    print(f"{'Epoch':<8}{'Total Errors':<16}{'E_average':<12}")
    print("-" * 36)
    for ep in range(1, MAX_EPOCHS + 1):
        total = 0
        for i in range(n):
            e = y[i] - step(dot(w, X[i]) + b)        # error = target - output
            total += abs(e)
            if e != 0:                               # perceptron learning rule
                for j in range(m):
                    w[j] += LEARNING_RATE * e * X[i][j]
                b += LEARNING_RATE * e
        e_avg = total / n                            # E_average = total errors / samples
        epochs.append(ep)
        e_avg_list.append(e_avg)
        acc_train.append(accuracy(X, y, w, b))
        acc_t1.append(accuracy(t1X, t1y, w, b))
        acc_t2.append(accuracy(t2X, t2y, w, b))
        print(f"{ep:<8}{total:<16}{e_avg:<12.4f}")
        if total == 0:
            print(f"\n>>> Converged at epoch {ep} <<<")
            break

    print("\nFinal weights:")
    for j, name in enumerate(FEATURES):
        print(f"  w[{name}] = {w[j]:.4f}")
    print(f"  bias = {b:.4f}")

    # ---- the 3 graphs ----
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 4.8))
    a1.bar(epochs, e_avg_list, color="#4C72B0")
    a1.set_title("Graph 1: E_average (per epoch)")
    a1.set_xlabel("Epoch"); a1.set_ylabel("E_average"); a1.grid(axis="y", linestyle="--", alpha=0.5)
    a2.plot(epochs, e_avg_list, marker="o", color="#D62728", linewidth=1.8)
    a2.set_title("Graph 2: E_average vs Epoch")
    a2.set_xlabel("Epoch"); a2.set_ylabel("E_average"); a2.grid(True, linestyle="--", alpha=0.5)
    a3.plot(epochs, acc_train, marker="o", color="#2CA02C", linewidth=2.2, label="Train accuracy")
    a3.plot(epochs, acc_t1, marker="s", color="#1F77B4", linewidth=1.5, linestyle="--", label="Test 1 accuracy")
    a3.plot(epochs, acc_t2, marker="^", color="#FF7F0E", linewidth=1.5, linestyle="--", label="Test 2 accuracy")
    a3.set_title("Graph 3: Accuracy vs Epoch")
    a3.set_xlabel("Epoch"); a3.set_ylabel("Accuracy (%)"); a3.set_ylim(0, 105)
    a3.grid(True, linestyle="--", alpha=0.5); a3.legend()
    plt.tight_layout()
    plt.savefig("slp_iris_graphs.png", dpi=150)
    print("\n[Saved] slp_iris_graphs.png")
    plt.show()
    return w, b


# ---------------- MODULE 2: TESTING ----------------
def test_slp(test_files, w, b):
    print("\n" + "=" * 60)
    print(f" MODULE 2 : TESTING  ({len(test_files)} test files)")
    print("=" * 60)
    for k, fname in enumerate(test_files, 1):
        X, y = read_csv(fname)
        correct = 0
        print(f"\nTest file {k}: {fname}  ({len(X)} samples)")
        print(f"{'Sample':<8}{'Net Input':<12}{'Pred':<6}{'Actual':<8}{'Status'}")
        print("-" * 44)
        for i in range(len(X)):
            net = dot(w, X[i]) + b
            p = step(net)
            ok = (p == y[i])
            correct += ok
            print(f"{i + 1:<8}{net:<12.3f}{p:<6}{y[i]:<8}{'CORRECT' if ok else 'WRONG'}")
        print("-" * 44)
        print(f"Accuracy = {correct}/{len(X)} = {100.0 * correct / len(X):.2f}%")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    prepare_files()
    w, b = train_slp(TRAIN_FILE, TEST_FILES)
    test_slp(TEST_FILES, w, b)