ImageMatcher

-> ImageMatcher is your GPU-powered visual search companion. Drop in an image, and it quickly surfaces the most visually similar pictures from your local dataset—no internet needed, just lightning-fast performance.

Key Features

-> Instant similarity search: Harnesses CLIP (ViT-B/32) to find top matches from your personalized image repository.

-> Compact, intuitive interface: Displays results as thumbnails with similarity scores and titles.

-> Efficient search logic: Uses precomputed embeddings for responsive matching even on large datasets.

How It Works
-> When you upload an image, CLIP generates a feature vector and compares it with your dataset’s stored embeddings. The server returns the best matches, which the interface neatly displays in real-time.

Tech Stack

-> Flask — REST-powered backend

-> CLIP (ViT-B/32) — for encoding and comparing images

-> Pandas + Pickle — for fast metadata and embedding storage
