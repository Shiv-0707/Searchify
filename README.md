# 🔍 Searchify - GPU-Powered Visual Search Engine

**Lightning-fast image similarity search using CLIP embeddings and cosine similarity matching**

Searchify is an intelligent visual search system that finds visually similar images from your local dataset using OpenAI's CLIP model. No internet required—just GPU-accelerated performance for instant results.

## ✨ Features

### Core Capabilities
- **CLIP-Powered Embeddings**: Uses ViT-B/32 model for robust image understanding
- **GPU Acceleration**: CUDA-optimized inference for lightning-fast results
- **Precomputed Embeddings**: Pre-process your dataset once, search instantly
- **Cosine Similarity**: Find truly similar images based on visual features
- **Thread-Safe**: Handle multiple concurrent requests safely
- **Real-time Results**: Returns matches with similarity scores

### Advanced Features
- **Metadata Support**: Store and retrieve image titles and metadata
- **Base64 Encoding**: Seamless image transmission and display
- **URL-Based Input**: Load images directly from URLs
- **Batch Processing**: Efficient handling of large datasets
- **Health Check Endpoint**: Monitor system status
- **Thumbnail Generation**: Automatic thumbnail creation (160x120)

## 🛠️ Installation

### Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- PyTorch
- Flask
- CLIP
- Pandas
- NumPy
- Pillow

### Setup

```bash
# Clone repository
git clone https://github.com/Shiv-0707/Searchify.git
cd Searchify

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Quick Start

### Running the Server

```bash
python app_clip_only.py
```

Server starts at `http://localhost:5000`

### Configuration

Edit `app_clip_only.py` to customize:

```python
IMAGE_DIR = r"C:\path\to\images"                    # Image dataset location
CLIP_MAP_PATH = r"C:\path\to\embeddings.pkl"         # Precomputed embeddings
META_FILE = r"C:\path\to\metadata.csv"              # Image metadata
CLIP_MODEL_NAME = "ViT-B/32"                         # CLIP model variant
THUMBNAIL_SIZE = (160, 120)                          # Thumbnail dimensions
HOST = 'localhost'                                    # Server host
PORT = 5000                                           # Server port
MAX_IMAGE_SIZE = 10 * 1024 * 1024                    # 10MB limit
```

## 📡 API Endpoints

### 1. Process Image (POST)

**Upload and search similar images**

```bash
curl -X POST http://localhost:5000/process \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,...", "threshold": 0.5, "max_results": 50}'
```

**Response:**
```json
{
  "category": "CLIP-match",
  "confidence": 1.0,
  "max_similarity": 0.95,
  "results": [
    {
      "category": "products",
      "file_name": "product_001.jpg",
      "image_name": "Red Shoes - Size 10",
      "title": "Red Shoes - Size 10",
      "similarity": 0.93,
      "image_data": "data:image/jpeg;base64,..."
    }
  ]
}
```

### 2. Process URL (POST)

**Download and search images from URL**

```bash
curl -X POST http://localhost:5000/process_url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.jpg"}'
```

### 3. Health Check (GET)

**Monitor system status**

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "clip_map_size": 5000
}
```

## 📊 Data Preparation

### Precompute Embeddings

```python
import clip
import torch
from PIL import Image
import pickle
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

embeddings_map = {}
image_dir = "your_image_directory"

for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)
        
        with torch.no_grad():
            image_input = preprocess(image).unsqueeze(0).to(device)
            embedding = model.encode_image(image_input).cpu().numpy()[0]
            embeddings_map[filename] = embedding

# Save embeddings
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings_map, f)
```

### Metadata CSV Format

```csv
image_name,category,title
product_001.jpg,shoes,Red Shoes - Size 10
product_002.jpg,shoes,Blue Shoes - Size 8
product_003.jpg,bags,Leather Bag - Brown
```

## 🔧 Technical Architecture

### Key Components

| Component | Purpose |
|-----------|----------|
| **CLIP Model** | ViT-B/32 for image encoding |
| **Flask** | REST API server |
| **Pickle** | Embedding serialization |
| **Pandas** | Metadata management |
| **Threading Lock** | Concurrent request safety |
| **NumPy** | Cosine similarity computation |

### How It Works

```
1. User uploads image
   ↓
2. Image preprocessed (resize, normalize)
   ↓
3. CLIP encodes to embedding vector
   ↓
4. Cosine similarity computed against all stored embeddings
   ↓
5. Top matches sorted by similarity
   ↓
6. Results returned with thumbnails and metadata
```

### Performance Characteristics

- **First Query**: ~2-3 seconds (model loading)
- **Subsequent Queries**: 100-500ms depending on dataset size
- **GPU Memory**: ~4GB for ViT-B/32
- **Dataset Capacity**: Limited by available RAM (tested up to 50k images)

## 📈 Usage Examples

### E-commerce Product Search

```bash
# Find similar products by image
curl -X POST http://localhost:5000/process \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,...",
    "threshold": 0.6,
    "max_results": 10
  }'
```

### Content-Based Image Retrieval

```bash
# Search by URL
curl -X POST http://localhost:5000/process_url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/reference.jpg"}'
```

## 🎯 Model Information

### ViT-B/32
- **Architecture**: Vision Transformer Base
- **Input Size**: 224×224 pixels
- **Output Dimension**: 512
- **Training Data**: 400M image-text pairs
- **Zero-Shot**: Excellent generalization

## ⚡ Performance Optimization

### Tips for Speed

1. **Batch Embeddings**: Precompute all embeddings once
2. **Use GPU**: CUDA support ~100x faster than CPU
3. **Adjust Threshold**: Higher threshold = fewer results to process
4. **Resize Images**: Preprocess images to 224×224
5. **Use Threading**: Enable thread locking for concurrent queries

### Memory Optimization

- Single ViT-B/32 embedding: ~2KB
- 10,000 images: ~20MB in memory
- 50,000 images: ~100MB in memory

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
- **Solution**: Reduce batch size or use CPU mode
- **Alternative**: Split embeddings computation into batches

### Issue: Slow inference
- **Solution**: Ensure CUDA is properly configured
- **Check**: Run `torch.cuda.is_available()`

### Issue: Bad similarity results
- **Solution**: Adjust threshold value
- **Tip**: Try threshold between 0.5-0.7

## 🔐 Limitations

- Single GPU support currently
- Images larger than 10MB rejected
- Embeddings must be precomputed offline
- No distributed inference

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

EXPOSE 5000
CMD ["python", "app_clip_only.py"]
```

### Deployment to Vercel (Serverless)

The project is deployed at: `https://searchify-taupe.vercel.app/`

## 📝 License

MIT License - Feel free to use for commercial projects

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Multi-GPU support
- Alternative models (ViT-L, ResNet-based)
- Incremental embedding updates
- GraphQL API
- Mobile app integration

## 📧 Contact

Questions or suggestions? Contact: Shiv Pratap Singh (Shiv-0707)

---

**Powered by OpenAI CLIP | Built with Flask | GPU-Accelerated**
