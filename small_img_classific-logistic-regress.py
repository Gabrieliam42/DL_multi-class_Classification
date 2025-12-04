# Script Developer: Gabriel Mihai Sandu
# GitHub Profile: https://github.com/Gabrieliam42

import os
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image, ImageFile, features

DATA_DIR = r"D:\AI_Research\DL\imgdata"
CHECKPOINT_DIR = r"D:\AI_Research\DL\imgcheckpoints"

BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_EPOCHS = 5
LR = 1e-3
VAL_SPLIT = 0.2
WORKERS = 4
PRINT_EVERY_BATCH = 20

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

ImageFile.LOAD_TRUNCATED_IMAGES = True


def log(msg: str):
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def find_image_files(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)


class FlatImageDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def prepare_datasets(data_dir):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    webp_ok = features.check("webp")
    if not webp_ok:
        log("WARNING: Pillow compiled without WEBP support. .webp files may fail to open.")
        log("If you see errors opening .webp files, install Pillow with webp or convert images to jpg.")

    subdirs = [p for p in data_path.iterdir() if p.is_dir()]
    has_subfolders_with_images = False
    classes = []

    if subdirs:
        for d in subdirs:
            imgs = find_image_files(d)
            if imgs:
                has_subfolders_with_images = True
                classes.append(d.name)

    if has_subfolders_with_images:
        log("Detected class subfolders. Using torchvision.datasets.ImageFolder.")
        transform = transforms.Compose([
            transforms.Resize(int(IMAGE_SIZE * 1.1)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(root=str(data_path), transform=transform)
        class_names = dataset.classes
        total_files = len(dataset)
    else:
        files = [p for p in data_path.iterdir()
                 if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if not files:
            files = find_image_files(data_path)
        if not files:
            raise FileNotFoundError(
                f"No image files found under {data_dir}. Supported extensions: {sorted(IMG_EXTS)}"
            )
        log(f"Detected flat image folder with {len(files)} image files. Creating single class dataset.")
        transform = transforms.Compose([
            transforms.Resize(int(IMAGE_SIZE * 1.1)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        dataset = FlatImageDataset(files, transform=transform)
        class_names = ["class0"]
        total_files = len(dataset)

    val_size = int(math.floor(VAL_SPLIT * total_files))
    train_size = total_files - val_size
    if val_size == 0:
        log("Validation split resulted in 0 samples; using train only run.")
        train_dataset = dataset
        val_dataset = None
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset, class_names

def build_model(num_classes: int, device: torch.device):
    in_features = 3 * IMAGE_SIZE * IMAGE_SIZE
    model = nn.Linear(in_features, num_classes)
    return model.to(device)

def save_checkpoint(state, epoch, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
    torch.save(state, path)
    log(f"Saved checkpoint: {path}")

def main():
    log(f"Current working directory: {os.getcwd()}")
    log(f"Using data folder: {DATA_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}  |  torch: {torch.__version__}  |  cuda build: {torch.version.cuda}  |  cuda available: {torch.cuda.is_available()}")

    log("Preparing datasets...")
    train_ds, val_ds, class_names = prepare_datasets(DATA_DIR)
    num_classes = len(class_names)
    log(f"Classes ({num_classes}): {class_names}")

    log("Building dataloaders...")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=WORKERS, pin_memory=True)
    val_loader = (DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=WORKERS, pin_memory=True)
                  if val_ds is not None else None)

    try:
        sample_x, sample_y = next(iter(train_loader))
        log(f"Sample batch shapes: images {tuple(sample_x.shape)} labels {tuple(sample_y.shape)}")
    except Exception as e:
        log(f"ERROR while fetching sample batch: {e}")
        raise

    model = build_model(num_classes, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model built: {model.__class__.__name__}  |  total params: {total_params:,}  trainable: {trainable:,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    log("Starting training loop...")
    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            xb = xb.view(xb.size(0), -1)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * xb.size(0)
            running_correct += (preds == yb).sum().item()
            running_total += xb.size(0)
            global_step += 1

            if batch_idx % PRINT_EVERY_BATCH == 0 or batch_idx == 1:
                batch_loss = running_loss / running_total
                batch_acc = running_correct / running_total
                log(f"Epoch {epoch}  batch {batch_idx}/{len(train_loader)}  loss(avg so far): {batch_loss:.4f}  acc(avg so far): {batch_acc:.4f}")

        epoch_loss = running_loss / running_total if running_total else 0.0
        epoch_acc = running_correct / running_total if running_total else 0.0
        epoch_time = time.time() - epoch_start
        log(f"Epoch {epoch} COMPLETE  loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}  time: {epoch_time:.1f}s")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    xb = xb.view(xb.size(0), -1)  # flatten
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    preds = logits.argmax(dim=1)
                    val_loss += loss.item() * xb.size(0)
                    val_correct += (preds == yb).sum().item()
                    val_total += xb.size(0)
            val_loss = val_loss / val_total if val_total else 0.0
            val_acc = val_correct / val_total if val_total else 0.0
            log(f"Validation  loss: {val_loss:.4f}  acc: {val_acc:.4f}  samples: {val_total}")

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "num_classes": num_classes,
            "class_names": class_names,
            "loss": epoch_loss,
            "acc": epoch_acc,
        }
        save_checkpoint(state, epoch, CHECKPOINT_DIR)

    log("Training finished. Final save completed.")

    model.eval()
    sample_x = sample_x.to(device)
    sample_x = sample_x.view(sample_x.size(0), -1)          # flatten

    sample_logits = torch.softmax(model(sample_x), dim=1)
    sample_logits = sample_logits.detach()                 # <--- added line

    for i, prob in enumerate(sample_logits.cpu().numpy()):
        probs = ", ".join(f"{p:.3f}" for p in prob)
        log(f"Sample {i:>3} – probs: {probs}")


if __name__ == "__main__":
    main()

# Script Developer: Gabriel Mihai Sandu
# GitHub Profile: https://github.com/Gabrieliam42