## Package Structure

```
protopred/
├── protopred/
│   ├── protopred_celery.py      # Main Celery task implementation
│   └── celery_worker_protopred.py  # Celery worker setup
├── pyproject.toml               # Project dependencies
├── Dockerfile.protopred         # Docker image definition
└── README.md                    # This file
```

### Testing Locally

You can test the core API function without Celery:

```python
from protopred.protopred_celery import query_protopred_api_json

result = query_protopred_api_json("CCO", models="model_phys:water_solubility")
print(result)
```
