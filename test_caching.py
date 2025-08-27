#!/usr/bin/env python3
"""
Test script to verify model caching works properly.
This simulates multiple requests in a web backend scenario.
"""

import time
from binoculars import Binoculars

def test_model_caching():
    """Test that models are loaded once and reused across instances."""
    print("=== Testing Binoculars Model Caching ===\n")
    
    # Use small models for testing
    observer_model = "Qwen/Qwen1.5-1.8B"
    performer_model = "Qwen/Qwen1.5-1.8B-Chat"
    
    sample_text = """This is a test text to verify that our caching system works correctly. 
    The models should only load once, and subsequent instances should reuse the cached models."""
    
    print("1. Creating first Binoculars instance (should load models)...")
    start_time = time.time()
    bino1 = Binoculars(
        observer_name_or_path=observer_model,
        performer_name_or_path=performer_model
    )
    first_load_time = time.time() - start_time
    print(f"First instance created in {first_load_time:.2f} seconds\n")
    
    print("2. Testing inference with first instance...")
    score1 = bino1.compute_score(sample_text)
    prediction1 = bino1.predict(sample_text)
    print(f"Score: {score1:.4f}, Prediction: {prediction1}\n")
    
    print("3. Creating second Binoculars instance (should use cached models)...")
    start_time = time.time()
    bino2 = Binoculars(
        observer_name_or_path=observer_model,
        performer_name_or_path=performer_model
    )
    second_load_time = time.time() - start_time
    print(f"Second instance created in {second_load_time:.2f} seconds\n")
    
    print("4. Testing inference with second instance...")
    score2 = bino2.compute_score(sample_text)
    prediction2 = bino2.predict(sample_text)
    print(f"Score: {score2:.4f}, Prediction: {prediction2}\n")
    
    print("5. Creating third instance with different mode...")
    start_time = time.time()
    bino3 = Binoculars(
        observer_name_or_path=observer_model,
        performer_name_or_path=performer_model,
        mode="accuracy"
    )
    third_load_time = time.time() - start_time
    print(f"Third instance created in {third_load_time:.2f} seconds\n")
    
    print("=== Results ===")
    print(f"First instance (fresh load): {first_load_time:.2f}s")
    print(f"Second instance (cached): {second_load_time:.2f}s") 
    print(f"Third instance (cached): {third_load_time:.2f}s")
    print(f"Speedup factor: {first_load_time / second_load_time:.1f}x")
    
    # Verify results are consistent (scores should be identical since models are cached)
    if isinstance(score1, list):
        score1 = score1[0]
    if isinstance(score2, list):
        score2 = score2[0]
    print(f"\nConsistent results: {abs(score1 - score2) < 1e-6}")
    
    print(f"\nCache contents:")
    print(f"Models cached: {len(Binoculars._model_cache)}")
    print(f"Tokenizers cached: {len(Binoculars._tokenizer_cache)}")

if __name__ == "__main__":
    test_model_caching()