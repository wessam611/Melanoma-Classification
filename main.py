from train import *
from test import *


def main():
    model = train()
    csv = predict(model, save_model=True, thresh=(0.7, 0.25), model_name=model)


if __name__ == "__main__":
    main()