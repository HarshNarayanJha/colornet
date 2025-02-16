import pickle

import numpy as np
import streamlit as st
from keras import saving
from keras.api.layers import TextVectorization
from keras.api.models import Sequential
from keras.api.utils import to_categorical

st.set_page_config(page_title="ColorNet")

st.title("Name to RGB Color Generation using RNNs")
name = st.text_input("Color Name")


@st.cache_data
def load_model() -> tuple[TextVectorization, Sequential]:
    try:
        model = saving.load_model("colornet.keras")
        if not isinstance(model, Sequential):
            raise TypeError("Loaded model is not of type Sequential")
    except Exception as e:
        st.error(e)
        st.stop()

    with open("tv.pkl", "rb") as fp:
        tv = pickle.load(fp)

    return tv, model


def scale(x: float) -> int:
    return int(x * 255)


def predict(name: str) -> np.ndarray:
    name = name.lower()
    tv, model = load_model()
    tokenized = tv([name])
    one_hot = to_categorical(tokenized.numpy(), num_classes=28)
    pred = model.predict(np.array(one_hot))[0]
    r, g, b = map(scale, pred)
    print(f"Predicted RGB:, {r}, {g}, {b}")
    return np.array([r, g, b])


if st.button("Generate Color"):
    color = predict(name)
    st.text(f"Generated Color is RGB({color})")
    st.image(np.full((32, 32, 3), color / 255), width=500)
