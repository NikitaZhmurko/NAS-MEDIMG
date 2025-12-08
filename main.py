import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score
from medmnist import INFO, TissueMNIST, PathMNIST
import numpy as np
import tqdm
from timeit import default_timer as timer
from tqdm.auto import tqdm

from static_model import StaticCNN  


# -----------------------------
# Training- & Evaluationsfunktionen
# -----------------------------

def train_epoch(model: torch.nn.Module, 
                data_loader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device):
    
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X, y in data_loader:
        y = y.squeeze().long()               # MedMNIST Labels [N,1] -> [N]
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        bs = X.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += bs

    avg_loss = total_loss / total_samples
    avg_acc  = (total_correct / total_samples) * 100.0
    return avg_loss, avg_acc


@torch.no_grad()
def eval_epoch(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X, y in data_loader:
        y = y.squeeze().long()               # MedMNIST Labels [N,1] -> [N]
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)

        bs = X.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += bs

    avg_loss = total_loss / total_samples
    avg_acc  = (total_correct / total_samples) * 100.0
    return avg_loss, avg_acc


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() 
    acc = (correct / len(y_pred)) * 100 
    return acc


# -----------------------------
# Main Function
# -----------------------------
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = device.type == "cuda"
    print(f"Using device: {device}")

    # -----------------------------
    # Dataset-Auswahl
    # -----------------------------
    if args.dataset == "tissuemnist":
        DatasetClass = TissueMNIST
    elif args.dataset == "pathmnist":
        DatasetClass = PathMNIST
    else:
        raise ValueError(f"Dataset {args.dataset} is not supported.")

    info = INFO[args.dataset.lower()]
    n_classes = len(info["label"])
    n_channels = info["n_channels"]

    # Kanal-abhängige Transforms
    if n_channels == 1:
        mean, std = [0.5], [0.5]
        as_rgb = False
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        as_rgb = True

    train_tf = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_set = DatasetClass(split="train", download=True, transform=train_tf, as_rgb=as_rgb)
    test_set  = DatasetClass(split="test",  download=True, transform=test_tf,  as_rgb=as_rgb)   

    # Validation Split
    val_size = int(0.1 * len(train_set))
    train_size = len(train_set) - val_size
    train_set, val_set = random_split(train_set, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(args.seed))

    train_dataloader = DataLoader(train_set, batch_size=args.batch, shuffle=True,  num_workers=0, pin_memory=pin_mem)
    val_dataloader   = DataLoader(val_set,   batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=pin_mem)
    test_dataloader  = DataLoader(test_set,  batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=pin_mem)
    
    print(f"Length of train dataloader: {len(train_dataloader)} batches of {args.batch}")
    print(f"Length of test dataloader: {len(test_dataloader)} batches of {args.batch}")
    print(f"Length of val dataloader: {len(val_dataloader)} batches of {args.batch}")

    # -----------------------------
    # Modell, Optimizer, Loss
    # -----------------------------
    model = StaticCNN(input_shape=n_channels, hidden_units=16, output_shape=n_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    # -----------------------------
    # Training Loop
    # -----------------------------

    epoch_train_loss, epoch_train_acc = [], []
    epoch_val_loss,   epoch_val_acc   = [], []

    best_val_loss = float("inf")
    best_path = f"best_{args.dataset.lower()}_cnn.pt"

    for ep in range(1, args.epochs + 1):
        # Training
        tr_loss, tr_acc = train_epoch(
            model=model,
            data_loader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        # Validation
        vl_loss, vl_acc = eval_epoch(
            model=model,
            data_loader=val_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        # Logging in Listen
        epoch_train_loss.append(tr_loss)
        epoch_train_acc.append(tr_acc)
        epoch_val_loss.append(vl_loss)
        epoch_val_acc.append(vl_acc)

        print(f"[{ep:02d}] "
            f"train loss {tr_loss:.4f} | acc {tr_acc:.2f}%  ||  "
            f"val loss {vl_loss:.4f} | acc {vl_acc:.2f}%")

        # Bestes Modell speichern (nach Val-Loss)
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ Saved best model ({best_path})")

    # --- finaler Test nach Training ---
    model.load_state_dict(torch.load(best_path, map_location=device))
    ts_loss, ts_acc = eval_epoch(model, test_dataloader, loss_fn, device)
    print(f"TEST  loss {ts_loss:.4f} | acc {ts_acc:.2f}%")

    np.savez(
        f"logs_{args.dataset.lower()}_iter.npz",
        epoch_train_loss=np.array(epoch_train_loss, dtype=np.float32),
        epoch_val_loss=np.array(epoch_val_loss, dtype=np.float32),
        epoch_train_acc=np.array(epoch_train_acc, dtype=np.float32),
        epoch_val_acc=np.array(epoch_val_acc, dtype=np.float32),
    )
    print("Saved metrics ->", f"logs_{args.dataset.lower()}_iter.npz")

# -----------------------------
# Argparse CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pathmnist", help="Dataset: tissuemnist | pathmnist")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    #parser.add_argument("--conv_layers", type=int, default=3)
    #parser.add_argument("--base_filters", type=int, default=32)
    #parser.add_argument("--kernel", type=int, default=3)
    #parser.add_argument("--dropout", type=float, default=0.3)
    #parser.add_argument("--dense", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
