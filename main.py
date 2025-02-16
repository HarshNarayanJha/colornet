# %% [md]
# # ColorNet

# %% [md]
# ## Importing Libraries

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential, layers
from keras.api.layers import TextVectorization
from keras.api.utils import to_categorical

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %%
data = pd.read_csv("./data/colors.csv")
print(data.head())
print(data.shape)
print(data.describe())

# %% [md]
# Let's check the distribution of name lengths

# %%
data["name"].map(len).plot(kind="hist", bins=20, density=True)

# %%
data[data["name"].map(len) > 25]
limited_data = data[data["name"].map(len) <= 25]

# %% [md]
# only 4 names have length greater than 25, and they are quite weird to be honest. We can ignore them.

# %%
limited_data["name"].map(len).sort_values()

# %% [md]
# # Preprocess Data
# We will tokenize the color names using keras text tokenization utility and pad the sequences to 25 len with zeroes.

# %%
max_len = 25
tv = TextVectorization(max_tokens=28, output_sequence_length=max_len, pad_to_max_tokens=True, split="character")
tv.adapt(limited_data["name"].values)
tokenized = tv(limited_data["name"].values)
print(limited_data["name"].values)
print(tokenized)


# %% [md]
# Now that the names are numerical, we can perform one hot encoding on them.

# %%
one_hot_names = to_categorical(tokenized.numpy())
print(one_hot_names.shape)

# %% [md]
# Now let's come to data normalization, the RGB values, must be between 0 and 1


# %%
def normalize(x) -> float:
    return x / 255.0


normalized_values = limited_data[["red", "green", "blue"]].apply(normalize)
normalized_values = np.column_stack([normalized_values["red"], normalized_values["green"], normalized_values["blue"]])
print(normalized_values)

# %% [md]
# Now let's do the model making/training

# %%
model = Sequential()
model.add(layers.Input(shape=(max_len, 28)))
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.LSTM(64))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(3, activation="sigmoid"))

model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

# %%
model.summary()


# %%
# Training
history = model.fit(one_hot_names, normalized_values, epochs=40, batch_size=32, validation_split=0.1)

# %%
df = pd.DataFrame(history.history)
ax = df.plot(figsize=(10, 6), title="Model Loss & Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss / Accuracy")
plt.legend()
plt.show()

# %%
# Save
model.save("colornet.keras")
model.save_weights("colornet.weights.h5")

# %% [md]
# ### Some prediction game!

# %%
# Global vars for the 3x3 plot
plot_idx = 0
fig = None
axes: np.ndarray | None = None


def plot_rgb(name: str, rgb: np.ndarray) -> None:
    global plot_idx, fig, axes

    if plot_idx % 9 == 0:
        if fig is not None:
            plt.show()
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    row = (plot_idx % 9) // 3
    col = (plot_idx % 9) % 3
    data = [[rgb]]
    if axes is not None:
        axes[row, col].imshow(data, interpolation="nearest")
        axes[row, col].axis("off")
        axes[row, col].set_title(name)

    plot_idx += 1

    if plot_idx % 9 == 0:
        plt.tight_layout(h_pad=0.5, w_pad=0.5)
        plt.show()


def scale(x: float) -> int:
    return int(x * 255)


def predict(color: str) -> None:
    name = color.lower()
    tokenized = tv([name])
    one_hot = to_categorical(tokenized.numpy(), num_classes=28)
    pred = model.predict(np.array(one_hot))[0]
    r, g, b = scale(pred[0]), scale(pred[1]), scale(pred[2])
    print(f"{color}: R,G,B: {r},{g},{b}")
    plot_rgb(f"{color} ({r},{g},{b})", pred)


# %%
predict("white")
predict("forest")
predict("sky")
predict("cerulean")
predict("crimson")
predict("turquoise")
predict("spring green")
predict("autumn red")
predict("summer blue")
predict("winter white")
predict("ocean blue")
predict("desert sand")
predict("mountain grey")
predict("jungle green")
predict("meadow green")
predict("sunset orange")
predict("dawn pink")
predict("dusk purple")
predict("cloud grey")
predict("storm grey")
predict("moss green")
predict("coral reef")
predict("arctic white")
predict("volcanic red")
predict("tropical green")
predict("prairie gold")
predict("black")
