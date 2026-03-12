# Searchify - GPU-Powered Visual Search Engine

## Lightning-fast image similarity search using CLIP embeddings and cosine similarity matching

Searchify is an intelligent visual search system that finds visually similar images from your local dataset using OpenAI's CLIP model. No internet required just GPU-accelerated performance for instant results.

## Features

### Core Capabilities

- **CLIP-Powered Embeddings**: Uses ViT-B/32 model for robust image understanding
- **GPU Acceleration**: CUDA-optimized inference for lightning-fast results
- **Precomputed Embeddings**: Pre-process your dataset once, search instantly
- **Cosine Similarity**: Find truly similar images based on visual features
- **Thread-Safe**: Handle multiple concurrent requests safely
- **Real-time Results**: Returns matches with similarity scores
- **Flexible Input**: Search by image file, URL, or base64 encoded image

### Advanced Features

- **Metadata Support**: Store and retrieve image titles and metadata
- **Base64 Encoding**: Seamless image transmission and display
- **URL-Based Input**: Load images directly from URLs
- **Batch Processing**: Efficient handling of large datasets
- **Health Check Endpoints**: Monitor system status
- **Thumbnail Generation**: Automatic thumbnail creation (160x120)

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- pip (Python package manager)
- Dependencies: torch, transformers, Pillow, FastAPI, uvicorn

### Setup

```bash
git clone https://github.com/Shiv-0707/Searchify.git
cd Searchify
pip install -r requirements.txt
```

## Quick Start

### Running the Server

```bash
python app.py
```

Server starts at http://localhost:8000

### Configuration

```python
# config.py
DATASET_PATH = "path/to/images"
METADATA_FILE = "metadata.csv"
SIMILARITY_THRESHOLD = 0.7
TOP_K_RESULTS = 5
BATCH_SIZE = 32
```

## API Endpoints

### 1. Process Image (POST)

Searches for similar images from a file upload

```
POST /search
Content-Type: multipart/form-data

Parameter: image (file upload)
Response: {"results": [{"image_path": "...", "similarity": 0.95}]}
```

### 2. Process URL (POST)

Searches using an image URL

```
POST /search-url
Content-Type: application/json

{"url": "https://example.com/image.jpg"}
Response: {"results": [{"image_path": "...", "similarity": 0.95}]}
```

### 3. Health Check (GET)

Verifies system status

```
GET /health
Response: {"status": "ok", "gpu_available": true}
```

## Data Preparation

### Precompute Embeddings

```bash
python precompute_embeddings.py --dataset-path /path/to/images --output embeddings.pkl
```

### Metadata CSV Format

```csv
image_path,title,category
images/photo1.jpg,Beach Sunset,Travel
images/photo2.jpg,Mountain Peak,Nature
```

## Technical Architecture

### Key Components

- **CLIP Encoder**: OpenAI's ViT-B/32 vision transformer
- **Embedding Storage**: Efficient NumPy array storage
- **Similarity Engine**: Cosine similarity computation
- **API Server**: FastAPI with async request handling
- **GPU Manager**: CUDA memory optimization

### How It Works

```
Input Image -> CLIP Encoder -> Embedding Vector
                                     |
                          Cosine Similarity Computation
                                     |
                          Top-K Matching Results
```

### Performance Characteristics

- **Encoding Speed**: ~50-100ms per image (GPU)
- **Search Speed**: <10ms for 10K images
- **Memory Usage**: ~512MB base + 4 bytes per embedding
- **Throughput**: 100+ queries/second

## Usage Examples

### E-commerce Product Search

```python
import requests

with open('product.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:8000/search', files=files)
    results = response.json()
    for match in results['results']:
        print(f"Match: {match['image_path']} ({match['similarity']:.2%})")
```

### Content-Based Image Retrieval

```python
response = requests.post(
    'http://localhost:8000/search-url',
    json={'url': 'https://example.com/query.jpg'}
)
print(response.json())
```

## Model Information

### ViT-B/32

- Vision Transformer Base variant
- 32-pixel patch size
- 384-dimensional embeddings
- Trained on 400M image-text pairs
- Multimodal alignment for semantic search

## Performance Optimization

### Tips for Speed

- Use GPU for encoding (10x faster than CPU)
- Precompute embeddings offline
- Use batch processing for multiple images
- Configure appropriate batch size (32-64 recommended)

### Memory Optimization

- Store embeddings in float32 (4 bytes each)
- Use memory-mapped files for large datasets
- Implement gradient checkpointing if fine-tuning

## Troubleshooting

### Issue: "CUDA out of memory"

- Reduce batch size
- Clear GPU memory: torch.cuda.empty_cache()
- Use CPU mode for smaller deployments

### Issue: Slow inference

- Ensure GPU is being used
- Check GPU utilization (nvidia-smi)
- Enable tensor optimization

### Issue: Bad similarity results

- Verify dataset preprocessing
- Check image quality and resolution
- Ensure metadata alignment

## Limitations

- Works best with natural images
- Requires sufficient GPU memory for batch processing
- CLIP model has inherent semantic understanding limits
- May not work well with artwork or abstract images

## Deployment

### Docker

```bash
docker build -t searchify .
docker run --gpus all -p 8000:8000 searchify
```

### Deployment to Vercel (Serverless)

Note: Serverless deployments are not recommended due to GPU requirements

## License

MIT License - Open source and free to use

## Contributing

Contributions welcome! Please submit pull requests or report issues.

## Contact

For questions or feedback: Shiv Pratap Singh (Shiv-0707)
