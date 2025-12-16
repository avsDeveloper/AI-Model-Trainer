# 🤖 ONNX Model Trainer v0.9

A complete GUI application for training transformer language models and exporting them to **GGUF** or **ONNX** format — from training to interactive chat testing, all in one place.

## ✨ What Makes This App Special

This tool provides an **end-to-end workflow**: select a model, train it on your dataset, export to your preferred format (GGUF or ONNX), and immediately test it in an interactive chat — all without leaving the application.

## 📦 Installation

### Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **Git** (for cloning llama.cpp)

### Step 1: Clone the Repository

```bash
git clone https://github.com/avsDeveloper/ONNX-Model-Trainer.git
cd ONNX-Model-Trainer
```

### Step 2: Install Python Dependencies

```bash
pip install torch transformers datasets numpy psutil accelerate
pip install onnx onnxruntime optimum
pip install onnxruntime-genai  # For ONNX model inference
```

**For GPU support (NVIDIA):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu
```

### Step 3: Set Up llama.cpp (Required for GGUF)

```bash
# Clone llama.cpp into the project directory
git clone https://github.com/ggerganov/llama.cpp.git

# Build llama.cpp
cd llama.cpp
make -j$(nproc)  # Linux/macOS
# or: cmake -B build && cmake --build build --config Release  # Windows
cd ..
```

### Step 4: Run the Application

```bash
python trainer.py
```

## 🖥️ Platform-Specific Notes

### Linux
```bash
sudo apt install python3-tk  # If tkinter is missing
```

### macOS
```bash
brew install python-tk
```

### Windows
- Python from python.org includes tkinter by default
- For llama.cpp, use Visual Studio or MinGW to build

## 📖 Understanding Model Formats

### GGUF (GPT-Generated Unified Format)
- **What it is**: A binary format designed for efficient CPU and GPU inference with llama.cpp
- **Best for**: Local deployment, edge devices, CPU inference, privacy-focused applications
- **Pros**: Small file sizes with quantization, fast inference, no Python required for deployment
- **Quantization options**: F16 (full precision), Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K (smallest)

### ONNX (Open Neural Network Exchange)
- **What it is**: An open format for representing machine learning models
- **Best for**: Cross-platform deployment, cloud services, ONNX Runtime integration
- **Pros**: Wide ecosystem support, hardware acceleration, framework interoperability
- **Quantization options**: QInt8, QUInt8 with Dynamic quantization

## ⚙️ Application Parameters

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Epochs** | Number of complete passes through the dataset | 3 |
| **Batch Size** | Samples processed before updating weights | 4 |
| **Learning Rate** | Step size for weight updates | 5e-5 |
| **Max Length** | Maximum token sequence length | 128 |
| **Save Steps** | Checkpoint save frequency | 500 |
| **Warmup Steps** | Gradual learning rate increase steps | 100 |
| **Scheduler** | Learning rate schedule (linear, cosine, constant) | linear |
| **Gradient Norm** | Maximum gradient magnitude for clipping | 1.0 |
| **Weight Decay** | L2 regularization strength | 0.01 |

### Training Presets

- **Quick Test**: Fast iteration for testing (1 epoch, small batch)
- **Balanced**: Good balance of speed and quality
- **Quality Focus**: Better results, longer training
- **Memory Saver**: For limited VRAM/RAM systems
- **Large Dataset**: Optimized for big datasets
- **Fine-tune**: Gentle training for pre-trained models

### Export Parameters (GGUF)

| Parameter | Description |
|-----------|-------------|
| **Quantization Type** | Compression level (F16 to Q2_K) |
| **Auto-fix EOS** | Automatically fix end-of-sequence token for chat models |

### Export Parameters (ONNX)

| Parameter | Description |
|-----------|-------------|
| **ONNX Opset** | ONNX operation set version (11-17) |
| **Quant Format** | Quantization format (QInt8/QUInt8) |
| **Quant Method** | Quantization method (Dynamic) |
| **Per-Channel** | Better quality quantization |
| **Reduce Range** | Better hardware compatibility |

### Generation Parameters (Testing)

| Parameter | Description | Range |
|-----------|-------------|-------|
| **Max Tokens** | Maximum response length | 1-2048 |
| **Temperature** | Randomness (lower = focused) | 0.0-2.0 |
| **Top-P** | Nucleus sampling threshold | 0.0-1.0 |
| **Top-K** | Number of top tokens to consider | 1-100 |
| **Repetition Penalty** | Penalize repeated tokens | 1.0-2.0 |

## 🔄 Workflow

### 1. Training Tab

1. **Select Base Model**: Choose from supported models (GPT-2, DialoGPT, Qwen, Phi, etc.)
2. **Choose Actions**: 
   - ☑️ Train — Fine-tune on your dataset
   - ☑️ Export — Convert to GGUF/ONNX
   - ☑️ Quantize — Compress the model
3. **Configure Training**: Select preset or customize parameters
4. **Browse Dataset**: Select your JSON dataset file
5. **Start Training**: Click "Start Training" or "Convert & Export"

### 2. Testing Tab

1. **Select Format**: Choose GGUF or ONNX
2. **Quick Select**: Pick from your exported models
3. **Choose Mode**: Chat, Q&A, Text Generation, etc.
4. **Chat**: Enter prompts and interact with your model

## 📁 Dataset Format

Create a JSON file with conversation pairs:

```json
[
  {"input": "Hello!", "output": "Hi there! How can I help you?"},
  {"input": "What's the weather?", "output": "I don't have weather data, but I can help with other questions."}
]
```

## 📂 Output Structure

```
output/
└── ModelName/
    ├── 1_trained/           # Trained model files
    │   ├── model.safetensors
    │   ├── config.json
    │   └── tokenizer.json
    └── 2_converted/         # Exported models
        ├── model.gguf       # GGUF format
        └── model.onnx       # ONNX format
```

## 🛠️ Supported Models

### Small Chat Models (Recommended)

| Model | Size | GGUF | ONNX | Notes |
|-------|------|------|------|-------|
| SmolLM 135M/360M/1.7B Instruct | 135M-1.7B | ✅ | ✅ | HuggingFace compact chat models |
| SmolLM2 135M/360M/1.7B Instruct | 135M-1.7B | ✅ | ✅ | Improved 2nd generation |
| Qwen2 0.5B/1.5B | 0.5B-1.5B | ✅ | ✅ | Alibaba's efficient models |
| Qwen2.5 0.5B/1.5B/3B Instruct | 0.5B-3B | ✅ | ✅ | Latest Qwen, excellent quality |
| TinyLlama 1.1B Chat | 1.1B | ✅ | ✅ | Compact Llama-based chat |
| StableLM 2 1.6B / Zephyr 1.6B | 1.6B | ✅ | ✅ | Stability AI chat models |
| StableLM Zephyr 3B | 3B | ✅ | ✅ | Larger StableLM variant |
| Phi-1/1.5/2 | 1.3B-2.7B | ✅ | ✅ | Microsoft code-focused |
| Phi-3 Mini 4K Instruct | 3.8B | ✅ | ✅ | Microsoft's latest compact |
| MiniCPM 2B | 2B | ✅ | ❌ | OpenBMB compact chat |

### Base/Legacy Models (Training & ONNX only)

| Model | Size | GGUF | ONNX | Notes |
|-------|------|------|------|-------|
| GPT-2 / DistilGPT-2 | 82M-1.5B | ❌ | ✅ | Great for learning |
| DialoGPT Small/Medium/Large | 117M-774M | ❌ | ✅ | Conversational |
| GPT-Neo 125M/1.3B | 125M-1.3B | ❌ | ✅ | Open source GPT |
| OPT 125M/350M | 125M-350M | ❌ | ✅ | Meta's open models |

### Larger Models

| Model | Size | GGUF | ONNX | Notes |
|-------|------|------|------|-------|
| Gemma 2B/7B | 2B-7B | ✅ | ✅ | Google's open models |
| Llama 2 7B | 7B | ✅ | ⚠️ | Requires authentication |
| Mistral 7B | 7B | ✅ | ⚠️ | High quality 7B |

## ❓ Troubleshooting

### "llama.cpp not found"
Build llama.cpp in the project directory:
```bash
cd llama.cpp && make -j$(nproc)
```

### "CUDA out of memory"
- Reduce batch size
- Use "Memory Saver" preset
- Enable gradient checkpointing

### "Model generates garbage"
- Ensure proper chat template is applied
- Check tokenizer files are preserved
- Try different generation parameters

## 📄 License

MIT License — Use at your own risk.

---

*This app was primarily written with AI assistance, intended to train AI models using AI-generated datasets. These models will be used by AI-based apps.* 🤖
