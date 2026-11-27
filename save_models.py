# File: save_models.py
# Kaam: Sirf ek baar chalana hai - Models ko Download karke ./models me Save karne ke liye

from sentence_transformers import SentenceTransformer

print("Saving Text Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('./models/text_model')
print("Text Model Saved - ./models/text_model")

print("Saving CLIP Model...")
clip_model = SentenceTransformer('clip-ViT-B-32')
clip_model.save('./models/clip_model')
print("CLIP Model Saved - ./models/clip_model")

print("Done! Ab models folder me save ho gaye.")