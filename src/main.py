import asyncio

from viam.module.module import Module

try:
    # Imported to register the model with Viam's resource registry.
    from src.models.classifier import Classifier  # noqa: F401
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.classifier import Classifier  # noqa: F401


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
