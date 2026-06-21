# Program 4 (Find-S)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Create dataset
df = pd.DataFrame(
{
    "Weather": ["Sunny", "Sunny", "Rainy", "Sunny"],
    "Temp": ["Warm", "Warm", "Cold", "Warm"],
    "Humidity": ["Normal", "High", "High", "High"],
    "Wind": ["Strong", "Strong", "Strong", "Weak"],
    "PlayTennis": ["Yes", "Yes", "No", "Yes"]
}
)

# Save CSV
df.to_csv("training.csv", index=False)

print("CSV file created successfully\n")


# ✅ Display CSV file
data = pd.read_csv("training.csv")

print("CSV file content:\n")
print(data)


# Take only positive examples
pos = data[data["PlayTennis"] == "Yes"].iloc[:, :-1].values


# Find-S hypothesis
final_hypothesis = np.where(
    pos.min(axis=0) == pos.max(axis=0),
    pos[0],
    "?"
)

print("\nFinal hypothesis:", final_hypothesis)