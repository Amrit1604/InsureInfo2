"""
🚀 SPEED OPTIMIZATION STRATEGIES
===============================
Target: Reduce 40s response to <15s

Current Bottlenecks:
1. Embeddings generation (batch_size=32 → 64)
2. Chunking strategy (reduce chunk count further)
3. API calls (faster models + parallel processing)
4. Document processing (smarter caching)
"""

def optimize_embeddings_speed():
    """Increase embedding batch size and use faster model"""
    return {
        'batch_size': 64,  # Double the current size
        'show_progress_bar': False,  # Remove progress overhead
        'convert_to_tensor': False,  # Use numpy directly
        'normalize_embeddings': True
    }

def optimize_chunking_strategy():
    """More aggressive chunking for speed"""
    return {
        'max_chunk_size': 2500,  # Larger chunks
        'overlap': 50,  # Minimal overlap
        'min_chunk_size': 500,  # Larger minimum
        'aggressive_merge': True
    }

def optimize_api_calls():
    """Faster API configuration"""
    return {
        'temperature': 0.0,  # Fastest generation
        'max_output_tokens': 500,  # Shorter responses
        'top_p': 0.9,
        'top_k': 20,  # Reduce computation
        'timeout': 15  # Shorter timeout
    }

def optimize_search_params():
    """Faster search configuration"""
    return {
        'top_k': 5,  # Fewer chunks for speed
        'similarity_threshold': 0.3,  # Higher threshold
        'max_search_k': 10  # Limit search space
    }

SPEED_OPTIMIZATIONS = {
    'embeddings': optimize_embeddings_speed(),
    'chunking': optimize_chunking_strategy(), 
    'api': optimize_api_calls(),
    'search': optimize_search_params()
}
