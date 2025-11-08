import numpy as np
import matplotlib.pyplot as plt

# === 1. Load the dataset ===
path = "sp500_regression.npz"  # change if it's in another folder
data = np.load(path)

print("Keys in file:", list(data.keys()))

# === 2. Inspect array shapes and types ===
for key in data.files:
    arr = data[key]
    print(f"\n🔹 {key}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"   min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}, std={arr.std():.4f}")

# === 3. Inspect one example sequence ===
X_train = data["X_train"]
y_train = data["y_train"]

i = 0  # you can change this index to explore others
seq = X_train[i].squeeze()  # shape (60,)
label = y_train[i]

print(f"\nExample {i}:")
print(f"Sequence length: {len(seq)}")
print(f"First 10 timesteps: {seq[:10]}")
print(f"Target (next normalized value): {label:.4f}")

# === 4. Plot the sequence and label ===
plt.figure(figsize=(8,4))
plt.plot(range(60), seq, marker='o', label='Past 60 normalized values')
plt.axhline(y=label, color='red', linestyle='--', label=f"Target y={label:.4f}")
plt.title("One Training Sequence and its Target (sp500_regression.npz)")
plt.xlabel("Time step (t-59 → t)")
plt.ylabel("Normalized value")
plt.legend()
plt.tight_layout()
plt.show()

# === 5. Optional: distribution of y values ===
plt.figure(figsize=(6,3))
plt.hist(y_train, bins=50, color='steelblue', edgecolor='white')
plt.title("Distribution of y_train (targets)")
plt.xlabel("Normalized target value")
plt.ylabel("Frequency")
plt.show()
