#!/usr/bin/env python3
"""
Example web backend using Flask to demonstrate model caching.
Models are loaded once at startup and reused across requests.
"""

from flask import Flask, request, jsonify
import time
from binoculars import Binoculars

app = Flask(__name__)

# Global detector instance - models loaded once at startup
print("Initializing Binoculars detector...")
start_time = time.time()
detector = Binoculars(
    observer_name_or_path="Qwen/Qwen1.5-1.8B",
    performer_name_or_path="Qwen/Qwen1.5-1.8B-Chat",
    mode="low-fpr"
)
load_time = time.time() - start_time
print(f"✓ Models loaded in {load_time:.2f} seconds")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_cached": len(Binoculars._model_cache),
        "tokenizers_cached": len(Binoculars._tokenizer_cache),
        "load_time": f"{load_time:.2f}s"
    })

@app.route('/detect', methods=['POST'])
def detect_ai_text():
    """Detect if text is AI-generated."""
    try:
        print("attempting to predict")
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({"error": "Empty text"}), 400
        
        # Process request using cached models
        start_time = time.time()
        score = detector.compute_score(text)
        prediction = detector.predict(text)
        processing_time = time.time() - start_time
        
        # Handle score type (can be float or list[float])
        # compute_score returns float for single string, list[float] for list of strings
        score_value = score if isinstance(score, (float, int)) else score[0]
        print("maybe success")
        return jsonify({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "score": score_value,
            "prediction": prediction,
            "processing_time": f"{processing_time:.3f}s",
            "threshold": detector.threshold,
            "mode": "low-fpr" if detector.threshold == 0.8536432310785527 else "accuracy"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/change_mode', methods=['POST'])
def change_detection_mode():
    """Change detection threshold mode."""
    try:
        data = request.get_json()
        if not data or 'mode' not in data:
            return jsonify({"error": "Missing 'mode' field"}), 400
        
        mode = data['mode']
        if mode not in ['low-fpr', 'accuracy']:
            return jsonify({"error": "Mode must be 'low-fpr' or 'accuracy'"}), 400
        
        detector.change_mode(mode)
        
        return jsonify({
            "mode": mode,
            "threshold": detector.threshold,
            "message": f"Mode changed to {mode}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear_cache', methods=['POST'])
def clear_model_cache():
    """Clear model cache (admin endpoint)."""
    try:
        Binoculars.clear_cache()
        return jsonify({"message": "Cache cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting web backend...")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /detect - Detect AI-generated text")
    print("  POST /change_mode - Change detection mode")
    print("  POST /clear_cache - Clear model cache")
    print(f"Server starting at http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)