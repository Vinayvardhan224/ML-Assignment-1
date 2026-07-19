import pandas as pd
import numpy as np

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")

X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
y = df["Payment (Rs)"].values

print("Dataset:\n")
print(df)
print("\nFeature Matrix (X):")
print(X)

print("\nOutput Vector (y):")
print(y)

print("\nDimensionality of the vector space:", X.shape[1])
print("Number of vectors:", X.shape[0])

rank = np.linalg.matrix_rank(X)
print("Rank of Feature Matrix:", rank)

cost = np.linalg.pinv(X) @ y

print("\nCost of each product:")
print("Candy        : ₹", round(cost[0], 2))
print("Mango (Kg)   : ₹", round(cost[1], 2))
print("Milk Packet  : ₹", round(cost[2], 2))


predicted = X @ cost

print("\nPredicted Payments:")
print(np.round(predicted, 2))

print("\nActual Payments:")
print(y)

comparison = pd.DataFrame({
    "Actual Payment": y,
    "Predicted Payment": np.round(predicted, 2)
})

print("\nComparison:")
print(comparison)