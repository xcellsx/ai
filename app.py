# app.py (Ensure this file exists and is correct)

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# --- Configuration ---
# *** IMPORTANT: Point this to the directory where trainer.save_model() saved the best model ***
MODEL_PATH = "./results/bert_uncased_L-2_H-128_A-2-finetuned-emotion/best_model" # Adjust if your path differs

# --- Validate Model Path ---
if not os.path.isdir(MODEL_PATH):
    print(f"---")
    print(f"FATAL ERROR: Model directory not found at the specified MODEL_PATH:")
    print(f"'{MODEL_PATH}'")
    print(f"---")
    print(f"Please ensure the path is correct relative to where you run 'python app.py'.")
    print(f"And that the training script successfully created the '{os.path.basename(MODEL_PATH)}' directory.")
    print(f"---")
    exit(1)

# --- Load Model and Tokenizer (Once on startup) ---
print(f"Loading model from: {MODEL_PATH}")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print("Model and Tokenizer loaded successfully.")

except Exception as e:
    print(f"--- FATAL ERROR loading model/tokenizer: {e} ---")
    exit(1)

# --- Emotion Labels (Ensure this matches your training data) ---
label_map = { 0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise" }
print(f"Using label map: {label_map}")

# --- Create Flask App ---
app = Flask(__name__)
CORS(app) # Enable CORS for requests from React dev server
print("Flask app created and CORS enabled.")

# --- API Endpoint for Analysis ---
@app.route('/analyze', methods=['POST'])
def analyze_text():
    print("\nReceived request for /analyze")
    try:
        if not request.is_json:
            print("Error: Request is not JSON")
            return jsonify({'error': 'Request must be JSON'}), 415

        data = request.get_json()
        if not data or 'text' not in data:
            print("Error: Invalid JSON or missing 'text' field")
            return jsonify({'error': 'Missing "text" field in JSON request body'}), 400

        text_input = data['text']
        if not isinstance(text_input, str) or not text_input.strip():
             print(f"Error: Received empty or invalid text: {text_input}")
             return jsonify({'error': 'Text input cannot be empty or invalid'}), 400

        print(f"Received text: '{text_input[:100]}...'")

        # --- Tokenize and Predict ---
        print("Tokenizing input...")
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=128) # Match training max_len

        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"Input tensors moved to device: {device}")

        print("Performing inference...")
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        emotion = label_map.get(predicted_class_id, f"unknown_id_{predicted_class_id}") # Handle unknown IDs

        print(f"Inference completed. Predicted ID: {predicted_class_id}, Label: {emotion}")

        # Return only the emotion label
        return jsonify({'emotion': emotion}), 200

    except Exception as e:
        print(f"--- UNEXPECTED SERVER ERROR during /analyze ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        # import traceback # Uncomment for full traceback during debugging
        # print(traceback.format_exc())
        print(f"--- End of Error ---")
        return jsonify({'error': 'An internal server error occurred during analysis'}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    print("--- Starting Flask Development Server ---")
    # Make sure host is '0.0.0.0' to be accessible from other devices/containers if needed
    # Port 5000 is the default
    app.run(host='0.0.0.0', port=10000, debug=False) # Set debug=False for production