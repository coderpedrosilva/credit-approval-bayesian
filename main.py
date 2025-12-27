from src.pipeline import run_pipeline
from train.train_model import train_and_persist

if __name__ == "__main__":
    run_pipeline()
    train_and_persist()
