import asyncio
from viam.module.module import Module


try:
    from models.classifier import Classifier
    print("MAIN _ A")
except ModuleNotFoundError:
    from .models.classifier import Classifier
    print("MAIN _ B")



if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
