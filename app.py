# app.py (Combined Emotion and Depression Detection)

import os
import sys # For potentially exiting on fatal errors
import traceback # For detailed error logging
from flask import Flask, request, jsonify
from flask_cors import CORS
# Ensure necessary transformers components are imported
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
import torch.nn as nn # Needed for the custom model definition
import numpy as np

# --- Configuration ---
# Path for the EMOTION model (Hugging Face format directory)
# This directory should contain config.json, pytorch_model.bin, tokenizer_config.json etc.
EMOTION_MODEL_PATH = "./results/best_model" # Adjust if your path differs

# Path for the DEPRESSION model weights (.pt file containing the state_dict)
DEPRESSION_MODEL_WEIGHTS = 'roberta_cnn_nf128_ks3_do40_lr1e-05_.pt' # Ensure this file is accessible

# --- Depression Model Definition (RobertaCNNClassifier) ---
# IMPORTANT: This class definition MUST exactly match the architecture
# used when the DEPRESSION_MODEL_WEIGHTS (.pt file) was saved.
class RobertaCNNClassifier(nn.Module):
    def __init__(self, model_name="roberta-base", num_filters=128, kernel_size=3, dropout=0.3):
        """
        Initializes the RobertaCNNClassifier model.

        Args:
            model_name (str): Name of the pre-trained RoBERTa model to use from Hugging Face.
            num_filters (int): Number of output channels for the Conv1D layer.
            kernel_size (int): Kernel size for the Conv1D layer.
            dropout (float): Dropout rate for the dropout layer.
        """
        super().__init__()
        # Using AutoModel to get the base RoBERTa model (embeddings and encoder layers)
        print(f"  [RobertaCNNClassifier] Initializing RoBERTa base: {model_name}")
        self.roberta = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.roberta.config.hidden_size # Get hidden size from RoBERTa config
        print(f"  [RobertaCNNClassifier] RoBERTa hidden size: {self.hidden_size}")

        # 1D Convolutional layer
        # Takes RoBERTa output (Batch, HiddenSize, SequenceLength)
        # Outputs (Batch, NumFilters, SequenceLength)
        self.conv1d = nn.Conv1d(in_channels=self.hidden_size,
                                out_channels=num_filters,
                                kernel_size=kernel_size,
                                padding=1) # padding=1 for kernel_size=3 approximates 'same' padding
        print(f"  [RobertaCNNClassifier] Conv1d initialized: {num_filters} filters, kernel size {kernel_size}")

        # Activation function
        self.relu = nn.ReLU()

        # Adaptive Max Pooling layer - pools across the sequence length dimension
        # Outputs (Batch, NumFilters, 1)
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        print(f"  [RobertaCNNClassifier] Dropout rate: {dropout}")

        # Fully connected output layer for 3 classes (Not Depressed, Moderate, Severe)
        self.fc = nn.Linear(num_filters, 3)
        print(f"  [RobertaCNNClassifier] Output layer initialized for 3 classes.")

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs (Batch, SequenceLength).
            attention_mask (torch.Tensor): Attention mask (Batch, SequenceLength).

        Returns:
            torch.Tensor: Logits for each class (Batch, 3).
        """
        # Pass input through RoBERTa base model
        # outputs.last_hidden_state shape: (Batch, SequenceLength, HiddenSize)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Prepare for Conv1D: needs (Batch, HiddenSize, SequenceLength)
        x = last_hidden_state.permute(0, 2, 1)

        # Apply Conv1D -> ReLU -> Pooling
        x = self.conv1d(x)       # (Batch, NumFilters, SequenceLength)
        x = self.relu(x)
        x = self.pool(x)         # (Batch, NumFilters, 1)

        # Remove the last dimension (SequenceLength dimension after pooling)
        x = x.squeeze(2)         # (Batch, NumFilters)

        # Apply Dropout and the final Fully Connected layer
        x = self.dropout(x)
        logits = self.fc(x)      # (Batch, 3)

        return logits

# --- Depression Prediction Function ---
# This function encapsulates the tokenization and prediction logic for the depression model
def predict_depression_multiclass(text, model, tokenizer, device='cpu', max_len=256):
    """
    Tokenizes text and predicts depression class using the provided model.

    Args:
        text (str): The input text.
        model (torch.nn.Module): The loaded RobertaCNNClassifier model instance.
        tokenizer: The loaded tokenizer instance (compatible with RoBERTa).
        device (torch.device): The device to run inference on ('cpu' or 'cuda').
        max_len (int): Maximum sequence length for tokenization.

    Returns:
        tuple: (predicted_class_id (int), probabilities (list))
    """
    print("  [Predict Depression] Tokenizing input...")
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length', # Pad to max_len
        truncation=True,      # Truncate if longer than max_len
        return_token_type_ids=False, # RoBERTa doesn't use token_type_ids
        return_attention_mask=True,  # Need attention mask
        return_tensors='pt'          # Return PyTorch tensors
    )

    # Move tensors to the specified device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Set model to evaluation mode and perform inference
    model.eval()
    print("  [Predict Depression] Performing inference...")
    with torch.no_grad(): # Disable gradient calculations for inference
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    # Calculate probabilities and predicted class
    probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item() # Get the index of the max probability

    print(f"  [Predict Depression] Logits: {logits.cpu().numpy()}")
    print(f"  [Predict Depression] Probabilities: {probs.cpu().numpy()}")
    print(f"  [Predict Depression] Predicted Class ID: {pred_class}")

    # Return class ID and the list of probabilities for all classes
    return pred_class, probs.squeeze().cpu().tolist() # Use cpu() before tolist()

# --- Model Loading ---
print("--- Initializing Models ---")
# Determine device (use CPU if CUDA is not available, typical for Render free tier)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize variables to None to track loading status
emotion_model, emotion_tokenizer = None, None
depression_model, depression_tokenizer = None, None
model_load_errors = [] # Keep track of errors during loading

# 1. Load Emotion Model (from Hugging Face directory format)
# ----------------------------------------------------------
print(f"\n[1/2] Loading EMOTION model from directory: {EMOTION_MODEL_PATH}")
# Check if the specified path is actually a directory
if not os.path.isdir(EMOTION_MODEL_PATH):
    error_msg = f"Emotion model directory not found at '{EMOTION_MODEL_PATH}'"
    print(f"ERROR: {error_msg}")
    model_load_errors.append(error_msg)
else:
    try:
        # Load tokenizer and model using Auto* classes from the directory
        emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)
        emotion_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_PATH)
        # Move model to device and set to evaluation mode
        emotion_model.to(device)
        emotion_model.eval()
        print("Emotion Model and Tokenizer loaded successfully.")
    except Exception as e:
        error_msg = f"Failed to load Emotion model/tokenizer from directory: {e}"
        print(f"ERROR: {error_msg}")
        print(traceback.format_exc()) # Print full traceback for debugging
        model_load_errors.append(error_msg)
        # Ensure variables are None if loading failed
        emotion_model, emotion_tokenizer = None, None

# 2. Load Depression Model (from .pt file using custom class)
# ------------------------------------------------------------
print(f"\n[2/2] Loading DEPRESSION model weights from file: {DEPRESSION_MODEL_WEIGHTS}")
# Check if the specified path is a file
if not os.path.isfile(DEPRESSION_MODEL_WEIGHTS):
    error_msg = f"Depression model weights file not found at '{DEPRESSION_MODEL_WEIGHTS}'"
    print(f"ERROR: {error_msg}")
    model_load_errors.append(error_msg)
else:
    try:
        # Load a standard RoBERTa tokenizer (assuming it's compatible)
        # If a specific tokenizer was fine-tuned/saved, load that instead.
        print("  Loading RoBERTa tokenizer for depression model...")
        depression_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        # Instantiate the custom model structure defined above
        # Ensure these hyperparameters match how the model was trained/saved
        print("  Instantiating RobertaCNNClassifier model structure...")
        depression_model = RobertaCNNClassifier(
            model_name="roberta-base",
            num_filters=128, # From filename 'nf128'
            kernel_size=3,   # From filename 'ks3'
            dropout=0.4      # From filename 'do40' -> 0.4
        )

        # Load the saved weights (state_dict) from the .pt file
        # map_location=device ensures compatibility between saving/loading devices
        print(f"  Loading state_dict from {DEPRESSION_MODEL_WEIGHTS}...")
        state_dict = torch.load(DEPRESSION_MODEL_WEIGHTS, map_location=device)

        # Load the weights into the model instance
        print("  Applying state_dict to model instance...")
        depression_model.load_state_dict(state_dict)

        # Move model to device and set to evaluation mode
        depression_model.to(device)
        depression_model.eval()
        print("Depression Model and Tokenizer loaded successfully.")
    except Exception as e:
        error_msg = f"Failed to load Depression model/tokenizer from .pt file: {e}"
        print(f"ERROR: {error_msg}")
        print(traceback.format_exc()) # Print full traceback for debugging
        model_load_errors.append(error_msg)
        # Ensure variables are None if loading failed
        depression_model, depression_tokenizer = None, None

# --- Final Check After Loading ---
if model_load_errors:
    print("\n--- WARNING: One or more models failed to load ---")
    for i, err in enumerate(model_load_errors):
        print(f"  Error {i+1}: {err}")
    print("--- Application will continue running, but affected predictions will fail. ---")
    # Optionally, you could exit if models are critical:
    # print("--- Exiting due to critical model load failure. ---")
    # sys.exit(1)
else:
    print("\n--- All models loaded successfully ---")


# --- Label Maps ---
# Emotion Labels (Ensure this matches your training data)
emotion_label_map = { 0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise" }
print(f"Using EMOTION label map: {emotion_label_map}")

# Depression Labels (Ensure this matches your training data)
depression_label_map = { 0: "Not Depressed", 1: "Moderate Depression", 2: "Severe Depression" }
print(f"Using DEPRESSION label map: {depression_label_map}")


# --- Create Flask App ---
app = Flask(__name__)
# Enable CORS for all domains on all routes. For production, you might want to restrict
# this to your specific Netlify frontend domain for security.
# Example: CORS(app, resources={r"/analyze": {"origins": "https://your-netlify-app.netlify.app"}})
CORS(app)
print("Flask app created and CORS enabled.")

# --- API Endpoint for Analysis ---
@app.route('/analyze', methods=['POST'])
def analyze_text():
    """
    Handles POST requests to /analyze.
    Performs emotion and depression analysis on the input text.
    Returns a JSON response with results for both models.
    """
    print("\nReceived request for /analyze")
    final_response = {} # Dictionary to store results
    status_code = 200   # Default HTTP status code

    try:
        # --- Input Validation ---
        if not request.is_json:
            print("Error: Request is not JSON")
            return jsonify({'error': 'Request content type must be application/json'}), 415

        data = request.get_json()
        if not data or 'text' not in data:
            print("Error: Invalid JSON or missing 'text' field")
            return jsonify({'error': 'Missing "text" field in JSON request body'}), 400

        text_input = data['text']
        if not isinstance(text_input, str) or not text_input.strip():
            print(f"Error: Received empty or invalid text: '{text_input}'")
            return jsonify({'error': 'Text input cannot be empty or invalid'}), 400

        print(f"Received text: '{text_input[:100]}...'") # Log beginning of text

        # --- 1. Emotion Analysis ---
        emotion_label = "N/A" # Default value
        # Check if the emotion model and tokenizer were loaded successfully
        if emotion_model and emotion_tokenizer:
            try:
                print("  [Analyze] Tokenizing for Emotion model...")
                # Tokenize input specifically for the emotion model
                inputs = emotion_tokenizer(
                    text_input,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128 # Ensure this matches model's expected max length
                )
                # Move input tensors to the correct device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                print("  [Analyze] Performing Emotion inference...")
                with torch.no_grad(): # Inference mode
                    outputs = emotion_model(**inputs)

                logits = outputs.logits
                predicted_class_id = torch.argmax(logits, dim=-1).item()
                # Map the predicted ID to its label string
                emotion_label = emotion_label_map.get(predicted_class_id, f"Unknown Emotion ID: {predicted_class_id}")
                print(f"  [Analyze] Emotion Inference completed. Label: {emotion_label}")

            except Exception as e:
                print(f"ERROR during Emotion prediction: {e}")
                print(traceback.format_exc()) # Log detailed error
                emotion_label = "Prediction Error"
        else:
            # Log if the model wasn't loaded
            print("  [Analyze] Emotion model not available for prediction.")
            emotion_label = "Model Not Loaded"
        # Add result to the final response dictionary
        final_response['emotion'] = emotion_label

        # --- 2. Depression Analysis ---
        depression_label = "N/A" # Default value
        # Check if the depression model and tokenizer were loaded successfully
        if depression_model and depression_tokenizer:
            try:
                print("  [Analyze] Predicting Depression...")
                # Use the dedicated prediction function
                pred_class, probs = predict_depression_multiclass(
                    text=text_input,
                    model=depression_model,
                    tokenizer=depression_tokenizer,
                    device=device,
                    max_len=256 # Use appropriate max length for RoBERTa/CNN model
                )
                # Map the predicted class ID to its label string
                depression_label = depression_label_map.get(pred_class, f"Unknown Depression Class: {pred_class}")
                print(f"  [Analyze] Depression Prediction completed. Label: {depression_label}, Probs: {probs}")

            except Exception as e:
                print(f"ERROR during Depression prediction: {e}")
                print(traceback.format_exc()) # Log detailed error
                depression_label = "Prediction Error"
        else:
            # Log if the model wasn't loaded
            print("  [Analyze] Depression model not available for prediction.")
            depression_label = "Model Not Loaded"
        # Add result to the final response dictionary
        final_response['depression'] = depression_label

        # --- Return Combined Results ---
        print(f"Sending response: {final_response}")
        return jsonify(final_response), status_code

    except Exception as e:
        # Catch-all for any unexpected errors during request processing
        print(f"--- UNEXPECTED SERVER ERROR in /analyze endpoint ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print(traceback.format_exc()) # Log detailed error
        print(f"--- End of Error ---")

        # Attempt to return a structured error response
        error_response = {'error': 'An internal server error occurred during analysis.'}
        # Include any partial results obtained before the error
        if 'emotion' in final_response: error_response['emotion'] = final_response.get('emotion', 'Error')
        if 'depression' in final_response: error_response['depression'] = final_response.get('depression', 'Error')
        return jsonify(error_response), 500


# --- Run the Flask App (for Local Development) ---
# This block is ignored when running with Gunicorn on Render.
# Render uses the 'Start Command' defined in its settings (e.g., 'gunicorn app:app').
if __name__ == '__main__':
    print("\n--- Starting Flask Development Server (for local testing only) ---")
    # Use port 5000 to match frontend's local target URL (http://localhost:5000)
    # host='0.0.0.0' makes it accessible from other devices on the same network
    # debug=False is recommended for stability, set to True for detailed error pages during dev
    app.run(host='0.0.0.0', port=5000, debug=False)
