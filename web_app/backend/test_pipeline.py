"""Quick test of the full inference pipeline."""
import sys
sys.path.insert(0, '.')
import numpy as np

from app.services.model_inference import model_service

print(f"Model loaded: {model_service.is_loaded()}")
print(f"Device: {model_service.get_device()}")
print(f"Labels: {len(model_service.get_label_map())}")

# Test with random data
dummy = np.random.randn(32, 543, 3).astype(np.float32)
result = model_service.predict_from_landmarks(dummy)

print(f"\nPrediction: {result['sign']} (conf={result['confidence']})")
print(f"Top5:")
for p in result['top5']:
    print(f"  {p['sign']}: {p['confidence']}")
print(f"Processing time: {result['processing_time_ms']}ms")
print("\n=== Pipeline test PASSED ===")
