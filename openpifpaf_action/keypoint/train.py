import argparse

import torch
import numpy as np
import random
from openpifpaf_action import keypoint


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def preprocess(anns):
    anns = [
        a
        for a in anns
        if ("actions" in a) and ("keypoints" in a) and (a["score"] > 0.01)
    ]
    inputs = []
    targets = []
    for a in anns:
        bbox = a["bbox"]
        kp = torch.Tensor(a["keypoints"]).reshape(-1, 3).float()
        kp = keypoint.transforms.normalize_with_bbox(kp, bbox, scale=True)
        label = torch.Tensor(a["action_labels"]).float()
        inputs.append(kp)
        targets.append(label)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return anns, inputs, targets


def shuffle(data):
    permutation = torch.randperm(len(data[0]))
    input = data[1][permutation]
    target = data[2][permutation]
    return input, target


def train_epoch(model, optimizer, criterion, data, batch_size, keypoint_drop):
    model.train(True)
    train_input, train_target = shuffle(data)
    epoch_loss = 0
    num_batches = 0
    for input, target in zip(
        train_input.split(batch_size), train_target.split(batch_size)
    ):
        if input.shape[0] == batch_size:
            input = keypoint.transforms.random_drop(input, keypoint_drop)
            input = input.reshape((batch_size, -1))
            prediction = model(input)
            loss = criterion(prediction, target)
            epoch_loss += loss.item()
            num_batches += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return epoch_loss / num_batches


def val_epoch(model, criterion, data, batch_size, keypoint_drop):
    model.eval()
    val_input, val_target = shuffle(data)
    epoch_loss = 0
    num_batches = 0
    for input, target in zip(val_input.split(batch_size), val_target.split(batch_size)):
        if input.shape[0] == batch_size:
            input = keypoint.transforms.random_drop(input, keypoint_drop)
            input = input.reshape((batch_size, -1))
            prediction = model(input)
            loss = criterion(prediction, target)
            epoch_loss += loss.item()
            num_batches += 1
    return epoch_loss / num_batches


def main(
    train_file,
    val_file,
    num_layers,
    linear_size,
    batchnorm,
    dropout,
    keypoint_drop,
    lr,
    seed,
    epochs,
    batch_size,
    save_path,
):
    set_seed(seed)
    train = keypoint.dataset.load_annotations(train_file)
    val = keypoint.dataset.load_annotations(val_file)
    train = preprocess(train)
    val = preprocess(val)
    model = keypoint.network.Baseline(
        num_layers=num_layers,
        input_size=np.prod(train[1].shape[1:]),
        output_size=train[2].shape[1],
        linear_size=linear_size,
        batchnorm=batchnorm,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    losses = []
    for epoch in range(epochs):
        train_loss = train_epoch(
            model, optimizer, criterion, train, batch_size, keypoint_drop
        )
        val_loss = val_epoch(model, criterion, val, batch_size, keypoint_drop)
        losses.append([train_loss, val_loss])
        print(
            f"Epoch {epoch:0>4} Train Loss {train_loss:>9.5f}, Val Loss {val_loss:>9.5}"
        )

    torch.save(model, save_path + ".model")
    torch.save(losses, save_path + ".losses")


parser = argparse.ArgumentParser("Keypoint Train")
parser.add_argument(
    "--train-file", type=str, required=True, help="Train annotations file"
)
parser.add_argument(
    "--val-file", type=str, required=True, help="Validation annotations file"
)
parser.add_argument(
    "--num-layers", type=int, required=True, help="Number of hidden layers"
)
parser.add_argument(
    "--linear-size", type=int, required=True, help="Width of hidden layers"
)
parser.add_argument(
    "--dropout",
    type=float,
    required=True,
    help="Dropout applied after each hidden layer",
)
parser.add_argument(
    "--keypoint-drop",
    type=float,
    required=True,
    help="Initial keypoint dropout, sets (x, y, score) = 0",
)
parser.add_argument("--lr", type=float, required=True, help="Learning rate")
parser.add_argument("--seed", default=0, type=int, help="Seed")
parser.add_argument("--epochs", type=int, required=True, help="Training epochs")
parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
parser.add_argument("--save-path", type=str, required=True, help="Output path")
parser.add_argument(
    "--batchnorm",
    default=False,
    action="store_true",
    help="Apply batchnorm after hidden layer",
)

if __name__ == "__main__":
    args = parser.parse_args()
    main(
        train_file=args.train_file,
        val_file=args.val_file,
        num_layers=args.num_layers,
        linear_size=args.linear_size,
        batchnorm=args.batchnorm,
        dropout=args.dropout,
        keypoint_drop=args.keypoint_drop,
        lr=args.lr,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_path,
    )
