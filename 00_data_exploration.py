# %% Imports
import matplotlib.pyplot as plt
from utils import DataLoader


# %% Load data
data_loader = DataLoader()
data_loader.load_dataset()

data = data_loader.data


# %% Show head
print(data.shape)
print(data.head())


# %% Show general statistics
data.info()


# %% Show histogram for all columns
columns = data.columns

for col in columns:
    print("Column:", col)

    plt.figure(figsize=(8, 5))

    data[col].hist()

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.show()
    plt.close()


# %% Show preprocessed dataframe
data_loader = DataLoader()
data_loader.load_dataset()

data_loader.preprocess_data()

print(data_loader.data.head())