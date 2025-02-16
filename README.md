# ColorNet: Generating RGB colors from text sequences using RNNs

A neural network that predicts RGB colors from text descriptions. Give it color names or descriptive phrases like "sunset orange" or "ocean blue", and it generates corresponding RGB values.

## How it Works

- Uses LSTM layers to process text input character by character
- Converts text descriptions into numerical sequences
- Outputs normalized RGB values between 0-1
- Trained on a dataset of color names and their RGB values

## Examples

Input -> Output (RGB)
- "ocean blue" -> (41, 128, 185)
- "forest" -> (34, 139, 34)
- "sunset orange" -> (253, 94, 83)
- "storm grey" -> (119, 136, 153)

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

Wanna try it out in your browser? Visit at https://colornet.streamlit.app/

## Requirements
- Python 3.7+
- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- Streamlit
