
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

data_list = []
xrays_dir = "data/xrays"

os.makedirs("data/CSVs", exist_ok=True)

for xray in sorted(os.listdir(xrays_dir)):
    xray_path = os.path.join(xrays_dir, xray)
    mask_path = os.path.join("data/masks", xray)

    if not os.path.isfile(xray_path):
        continue

    if not os.path.exists(mask_path):
        mask_path = None

    data_list.append((xray_path, mask_path))

data_df = pd.DataFrame(data_list, columns=["xrays", "masks"])
data_df.to_csv("data/CSVs/dataset.csv", index=False)

test_df = data_df[data_df["masks"].isna()]
labeled_df = data_df[data_df["masks"].notna()]

train_df, val_df = train_test_split(
    labeled_df, test_size=0.2, random_state=42
)

train_df.to_csv("data/CSVs/train.csv", index=False)
val_df.to_csv("data/CSVs/val.csv", index=False)
test_df.to_csv("data/CSVs/test.csv", index=False)

labels = ["Dataset", "Train", "Validation", "Test"]
counts = [len(data_df), len(train_df), len(val_df), len(test_df)]

plt.bar(labels, counts)
plt.xlabel("Dataset Split")
plt.ylabel("Number of Images")
plt.title("Dataset Distribution")

for i, count in enumerate(counts):
    plt.text(i, count + 2, str(count), ha="center")

plt.ylim(0, max(counts) + 20)
plt.savefig("data/CSVs/dataset_distribution.png",
            dpi=300, bbox_inches="tight")
plt.show()

print("Dataset:", len(data_df))
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))
