# app_clip_only.py
import os, io, pickle, base64, logging, signal, sys, requests
import torch, clip
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from io import BytesIO
import threading

# ---------------- CONFIG ----------------
IMAGE_DIR = r"C:\Users\shivp\Downloads\IMAGE_MATCHER\amazon_images"
CLIP_MAP_PATH = r"C:\Users\shivp\Downloads\IMAGE_MATCHER\amazon_clip\clip_embeddings.pkl"
META_FILE = r"C:\Users\shivp\Downloads\IMAGE_MATCHER\amazon_metadata\metadata.csv"
CLIP_MODEL_NAME = "ViT-B/32"
THUMBNAIL_SIZE = (160, 120)
HOST = 'localhost'
PORT = 5000
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB limit
# ----------------------------------------

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
CORS(app)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading CLIP on {device}...")

# Lock for safe inference
model_lock = threading.Lock()

# Load CLIP model
try:
    clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    clip_model.eval()
    logger.info("CLIP model loaded!")
except Exception as e:
    logger.error(f"Error loading CLIP: {e}")
    clip_model, clip_preprocess = None, None

# Load CLIP embeddings
try:
    with open(CLIP_MAP_PATH, "rb") as f:
        clip_map = pickle.load(f)
    logger.info(f"Loaded CLIP map with {len(clip_map)} embeddings")
except Exception as e:
    logger.error(f"Error loading CLIP map: {e}")
    clip_map = []

# Load metadata for titles
meta_titles = {}
try:
    df = pd.read_csv(META_FILE)
    for _, row in df.iterrows():
        key = (row["image_name"], row["category"])
        meta_titles[key] = row.get("title", "")
    logger.info(f"Loaded {len(meta_titles)} metadata entries")
except Exception as e:
    logger.error(f"Error loading metadata: {e}")


# ---------- Utils ----------

def get_embedding(pil_img):
    """Encode image into CLIP embedding"""
    if not clip_model:
        raise Exception("CLIP not loaded")

    img = clip_preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(img).cpu().numpy()[0]
    return emb / np.linalg.norm(emb)


def load_thumbnail_base64(entry):
    """Load thumbnail and return base64 string"""
    img_path = os.path.join(IMAGE_DIR, entry["category"].replace(" ", "_"), entry["image_name"])
    if not os.path.exists(img_path):
        return None
    try:
        im = Image.open(img_path).convert("RGB")
        im.thumbnail(THUMBNAIL_SIZE)
        buf = BytesIO()
        im.save(buf, format="JPEG")
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        return None


def process_image(image_data, threshold=0.5, max_results=50):
    """Find similar images with CLIP"""
    try:
        # Convert base64 → PIL
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        pil_img = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")

        with model_lock:
            query_emb = get_embedding(pil_img)

        # Similarities to all dataset embeddings
        sims = [(e, float(np.dot(query_emb, e["embedding"]))) for e in clip_map]
        sims.sort(key=lambda x: x[1], reverse=True)

        max_sim = sims[0][1] if sims else 0.0
        filtered = [(e, s) for e, s in sims if s >= threshold][:max_results]

        response = {
            "category": "CLIP-match",
            "confidence": 1.0,
            "max_similarity": max_sim,
            "results": []
        }

        for e, score in filtered:
            image_name = e.get("image_name", "")
            category = e.get("category", "")
            # Title from metadata
            title = e.get("title", "") or meta_titles.get((image_name, category), "")
            display_name = title if title else image_name

            thumb = load_thumbnail_base64(e)
            if not thumb:
                continue

            response["results"].append({
                "category": category,
                "file_name": image_name,      # actual filename
                "image_name": display_name,   # user sees title here
                "title": title,
                "similarity": score,
                "image_data": thumb
            })

        return response

    except UnidentifiedImageError:
        return {"error": "Invalid image format"}
    except Exception as e:
        logger.error(f"process_image failed: {e}")
        return {"error": str(e)}


# ---------- Routes ----------

@app.route('/')
def index():
    return render_template('index.html', device=device)


@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data"})

        threshold = float(data.get("threshold", 0.5))
        max_results = int(data.get("max_results", 50))

        result = process_image(data["image"], threshold, max_results)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/process_url', methods=['POST'])
def process_url():
    """Download image from URL and return base64"""
    try:
        data = request.get_json()
        if not data or "url" not in data:
            return jsonify({"error": "No URL provided"})

        url = data["url"].strip()
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()

        # Content check
        if "image" not in resp.headers.get("Content-Type", ""):
            return jsonify({"error": "URL does not point to an image"})

        if len(resp.content) > MAX_IMAGE_SIZE:
            return jsonify({"error": "Image too large (max 10MB)"})

        # Convert to base64
        img_b64 = base64.b64encode(resp.content).decode("utf-8")
        image_data = f"data:{resp.headers['Content-Type']};base64,{img_b64}"

        return jsonify({
            "success": True,
            "image_data": image_data,
            "message": "Image loaded successfully"
        })

    except Exception as e:
        logger.error(f"process_url failed: {e}")
        return jsonify({"error": str(e)})


@app.route('/health')
def health():
    status = {
        "status": "ok" if clip_model and clip_map else "fail",
        "device": device,
        "clip_map_size": len(clip_map)
    }
    return jsonify(status)


def shutdown_handler(sig, frame):
    logger.info("Shutting down...")
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    if not os.path.exists("templates/index.html"):
        with open("templates/index.html", "w") as f:
            f.write("<h1>Frontend goes here</h1>")
    logger.info(f"Starting server at http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)
