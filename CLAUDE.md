# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation & Setup

```bash
# Install the package in development mode
pip install -e .

# Install from requirements.txt
pip install -r requirements.txt
```

## Core Architecture

Binoculars is an AI-generated text detection system that uses two language models to compute perplexity-based scores for classification. The system implements **model caching** at the class level to optimize for web backend scenarios where models should be loaded once and reused across multiple requests.

### Core Detection Algorithm (`binoculars/detector.py`)
- `Binoculars` class: Main detector that loads two models (observer and performer)
- **Model Caching**: Models and tokenizers are cached at class level using `_model_cache` and `_tokenizer_cache`
- Uses dual-model architecture: typically a base model (observer) and instruct model (performer)  
- Default models: `tiiuae/falcon-7b` (observer) and `tiiuae/falcon-7b-instruct` (performer)
- Detection works by computing ratio of perplexity to cross-perplexity between models
- Two detection thresholds: "low-fpr" (optimized for low false positives) and "accuracy" (optimized for F1)
- Cache management: `Binoculars.clear_cache()` method to free memory when needed

### Supporting Modules
- `binoculars/metrics.py`: Implements perplexity and cross-entropy calculations
- `binoculars/utils.py`: Tokenizer consistency checks and utilities
- `binoculars/__init__.py`: Package exports

### Applications
- `main.py`: Simple CLI demo using smaller Qwen models (Qwen1.5-1.8B variants)
- `app.py`: Launches Gradio web interface 
- `demo/demo.py`: Gradio UI implementation with mode switching and validation
- `web_backend_example.py`: Flask web API demonstrating model caching for production use
- `test_caching.py`: Test script to verify model caching performance

## Common Commands

```bash
# Run CLI demo
python main.py

# Launch web interface
python app.py

# Install package for development
pip install -e .

# Test model caching functionality
python test_caching.py

# Run web backend example
python web_backend_example.py
```

## GPU Configuration

The detector automatically handles multi-GPU setups:
- Observer model loads on `cuda:0` or CPU fallback
- Performer model loads on `cuda:1` if available, otherwise shares `cuda:0`
- Supports CPU-only operation when CUDA unavailable

## Model Configuration

Models can be customized in the `Binoculars` constructor:
- `observer_name_or_path`: HuggingFace model path for observer model
- `performer_name_or_path`: HuggingFace model path for performer model  
- `use_bfloat16`: Enable bfloat16 precision for faster inference
- `max_token_observed`: Token limit per input (default 512)
- `mode`: Detection threshold ("low-fpr" or "accuracy")

For private HuggingFace models, set `HF_TOKEN` environment variable.

## Detection Workflow

1. Input text is tokenized using observer model's tokenizer
2. Both models generate logits for the input
3. Perplexity calculated under performer model
4. Cross-perplexity calculated between observer and performer logits
5. Binoculars score = perplexity / cross-perplexity
6. Classification based on threshold comparison

## Dataset Structure

- `datasets/core/`: Core evaluation datasets (cc_news, cnn, pubmed) with model outputs
- `datasets/robustness/`: Robustness testing datasets with varied prompting styles
- Data format: JSONL files with model-specific naming conventions

## Key Dependencies

- `transformers[torch]`: HuggingFace model loading and inference
- `torch`: PyTorch backend
- `gradio`: Web interface framework
- `numpy`: Numerical operations
- `scikit-learn`, `seaborn`, `pandas`: Analysis and visualization support