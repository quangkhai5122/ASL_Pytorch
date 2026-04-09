"""Quick test to check OpenAPI schema generation."""
import sys
sys.path.insert(0, '.')

from app.main import app
from fastapi.openapi.utils import get_openapi

try:
    schema = get_openapi(
        title="test",
        version="1.0",
        routes=app.routes
    )
    print("OpenAPI schema generated successfully!")
    print(f"Paths: {list(schema.get('paths', {}).keys())}")
except Exception as e:
    print(f"OpenAPI ERROR: {e}")
    import traceback
    traceback.print_exc()
