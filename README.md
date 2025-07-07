# 🤖 Mini-GPT: A Transformer-based Language Model from Scratch

This project implements a Transformer model, affectionately named "Mini-GPT," from the ground up using PyTorch. It's inspired by the original "Attention Is All You Need" paper by Vaswani et al. (2017) and focuses on the decoder-style architecture for language modeling tasks.

The primary goal of this project is to understand the inner workings of a Transformer by building each component:
* Multi-Head Self-Attention (MHSA)
* Position-wise Feedforward Networks (FFN)
* Positional Embeddings
* Encoder/Decoder Blocks (though this implementation focuses on a decoder-style model)

The model is trained to predict the next token in a sequence, using a custom text dataset.

---

## ✨ Features

-  **Transformer Implementation from Scratch**: All key components of the Transformer architecture are custom-built.
-  **Subword-Level Tokenization**: Uses `tiktoken` (the tokenizer used by GPT models) for efficient text processing.
-  **Configurable Hyperparameters**: Easily adjust `block_size`, `emb_dim`, `num_heads`, `num_layers`, etc., via a `Config` dataclass.
-  **Multi-Head Self-Attention**: Implements the core attention mechanism found in GPT models.
-  **Feedforward Network**: Includes a custom feedforward module with GELU activation and dropout.
-  **Pre-LN Layer Normalization**: Applies Layer Normalization before skip connections for better training stability.
-  **Language Modeling Task**: Trained to predict the next token and generate coherent text.
-  **Text Generation**: Includes a `generate()` method to produce new text sequences from a prompt.
-  **Parameter Analysis**: *(Planned/Included)* Visualizations to explore how hyperparameters affect model size and performance.

---

## 📁 File Structure

| File | Description |
|------|-------------|
| `Part_1.ipynb` | Core model implementation: embedding layers, attention blocks, decoder stack |
| `Part_2.ipynb` | Model training, text generation, and evaluation on sample dataset |

---

## 🛠️ Tech Stack

* **Python**
* **PyTorch:** For building and training the neural network.
* **tiktoken:** For subword tokenization.
* **NumPy:** For numerical operations.
* **Matplotlib:** (Optional, for parameter analysis plots)

---

## 📚 Project Structure & Components

The model is built modularly:

1.  **Dataset Preparation (`get_batch` function):**
    * Loads a custom text dataset (e.g., the "Kung Fu Panda" script in the example).
    * Uses `tiktoken` for encoding the text.
    * Generates batches of input sequences (`x`) and target sequences (`y`).

2.  **Configuration (`Config` dataclass):**
    * Manages hyperparameters like `block_size`, `emb_dim`, `head_size`, `num_heads`, `num_layers`, and `vocab_size`.

3.  **Multi-Head Self-Attention (`MHSA` class):**
    * Projects input into Query (Q), Key (K), and Value (V).
    * Computes attention scores using scaled dot-product attention.
    * Applies a causal mask for language modeling.
    * Concatenates outputs from multiple attention heads.

4.  **Feedforward Network (`Feedforward` class):**
    * Consists of three linear layers with GELU activations and a dropout layer.
    * Upsamples and then downsamples the dimensionality.

5.  **Transformer Block (`Block` class):**
    * Combines an `MHSA` module and a `Feedforward` module.
    * Uses Layer Normalization (Pre-LN design) and residual (skip) connections.

6.  **Mini-GPT Model (`SastaGPT` or `MiniGPT` class in your code):**
    * **Embeddings:**
        * `nn.Embedding` for token embeddings.
        * `nn.Embedding` for positional embeddings.
    * **Transformer Layers:** A sequence of `Block` modules.
    * **Output Layer:** A final Layer Normalization followed by a Linear layer to produce logits over the vocabulary.
    * **Weight Initialization:** Custom initialization for `Linear` and `Embedding` layers.
    * **`generate()` method:** For autoregressive text generation using the trained model.

    *(Note: You might want to rename the `SastaGPT` class in your Python code to `MiniGPT` for consistency if you prefer.)*

---

## ⚙️ Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder-name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch tiktoken numpy matplotlib
    ```
    *(Ensure you have PyTorch installed according to your CUDA version if using a GPU. Visit [pytorch.org](https://pytorch.org/) for instructions.)*

3.  **Prepare your dataset:**
    * Place your custom text file (e.g., `my_dataset.txt`) in the project directory.
    * Update the `file_path` variable in the notebook/script to point to your dataset.
        ```python
        # Example from the notebook:
        # file_path = r'C:\\Users\\Areeba Shahzad\\Desktop\\ML\\PA4\\kungfu_panda.txt'
        file_path = 'your_dataset.txt' # Modify this
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        text_array = np.array(list(text))
        ```

4.  **Configure Hyperparameters (Optional):**
    * Modify the `Config` dataclass in the script if you want to experiment with different model sizes or settings.
        ```python
        @dataclass
        class Config:
            block_size: int = BLOCK_SIZE # e.g., 8
            emb_dim: int = 256
            head_size: int = 32
            num_heads: int = 8
            num_layers: int = 2
            vocab_size: int = encoder.n_vocab # From tiktoken
        ```

5.  **Train the model:**
    * Run the training script/notebook cells. The training loop initializes the `MiniGPT` model, optimizer, and loss function, then iterates for a specified number of steps.
    * The trained model weights will be saved (e.g., to `minigpt_model.pth`).

6.  **Generate Text:**
    * Load the trained model weights if necessary.
    * Use the `model.generate()` method with a starting prompt.
        ```python
        # Ensure 'encoder' (tiktoken.get_encoding("gpt2")) is defined
        # Ensure 'model' is your trained MiniGPT model, set to eval mode
        # model.load_state_dict(torch.load('minigpt_model.pth')) # If loading saved model
        # model.eval()

        sentence = "Your starting prompt here"
        idxs = torch.tensor(encoder.encode(sentence)).unsqueeze(0).to(device) # Move to device if using GPU

        generated_tokens = model.generate(idxs, max_new_tokens=100)
        generated_text = encoder.decode(generated_tokens[0].cpu().numpy()) # Move to CPU before decoding
        print(generated_text)
        ```

---
