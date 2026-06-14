# uv run src/train.py

import pickle
from pathlib import Path
import sys
import traceback

# package-aware imports: allow running as script or as package
try:
    from src.load_fashion_mnist import load_train_data  # type: ignore
except Exception:
    from load_fashion_mnist import load_train_data

try:
    from src.network import NetworkConfig, SimpleMLP  # type: ignore
except Exception:
    from network import NetworkConfig, SimpleMLP

OUTPUT_PATH = Path("sample_weight.pkl")
EPOCHS = 20
HIDDEN_SIZE = 256
LEARNING_RATE = 0.01
BATCH_SIZE = 128
SEED = 42


def main() -> int:
    (x_train, t_train), (x_valid, t_valid) = load_train_data()

    model = SimpleMLP(
        NetworkConfig(
            input_size=x_train.shape[1],
            hidden_size=HIDDEN_SIZE,
            output_size=10,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            seed=SEED,
        )
    )

    for epoch in range(1, EPOCHS + 1):
        loss = model.train_epoch(x_train, t_train, epoch=epoch)
        train_acc = model.evaluate_accuracy(x_train, t_train)
        valid_acc = model.evaluate_accuracy(x_valid, t_valid)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"loss={loss:.4f} train_acc={train_acc:.4f} valid_acc={valid_acc:.4f}"
        )

    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(model.to_state(), f)

    print(f"Saved model: {OUTPUT_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
