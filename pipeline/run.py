from architecture.train import Trainer
from architecture.inference import Inferencer

PATH = "config.yaml"

if __name__ == "__main__":
    Trainer(PATH).run()
    Inferencer(PATH).run()