from architecture.train import Trainer
from architecture.inference import Inferencer

if __name__ == "__main__":
    Trainer().run()
    Inferencer().run()