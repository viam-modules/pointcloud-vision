import asyncio
from viam.module.module import Module

try:
    from src.models.classifier import Classifier
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.classifier import Classifier


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
