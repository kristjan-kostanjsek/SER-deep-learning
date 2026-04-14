import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Emotion mapping
EMOTIONS = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}

# ==========================================
# DATASET
# ==========================================
class CremaDataset(Dataset):
    def __init__(
        self,
        file_paths,
        is_train=False,
        use_augmentation=False,
        add_noise_std=0.0,
        freq_mask_param=0,
        time_mask_param=0
    ):
        self.file_paths = file_paths
        self.is_train = is_train
        self.use_augmentation = use_augmentation

        # Augmentation params
        self.add_noise_std = add_noise_std
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        spec = np.load(file_path)
        spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

        # AUGMENTATION (SpecAugment)
        if self.use_augmentation and self.is_train:
            n_mels, n_steps = spec.shape

            # Frequency masking
            if self.freq_mask_param > 0 and n_mels > self.freq_mask_param:
                f_mask = np.random.randint(0, self.freq_mask_param)
                f0 = np.random.randint(0, n_mels - f_mask)
                spec[f0:f0 + f_mask, :] = 0

            # Time masking
            if self.time_mask_param > 0 and n_steps > self.time_mask_param:
                t_mask = np.random.randint(0, self.time_mask_param)
                t0 = np.random.randint(0, n_steps - t_mask)
                spec[:, t0:t0 + t_mask] = 0

        # Convert to tensor
        spec = np.expand_dims(spec, axis=0)
        spec_tensor = torch.tensor(spec, dtype=torch.float32)

        # ✅ RESIZE (FOR DINO / ViT)
        import torch.nn.functional as F

        spec_tensor = F.interpolate(
            spec_tensor.unsqueeze(0),   # (1, C, H, W)
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # back to (C, H, W)

        # ✅ NORMALIZATION (IMPORTANT for DINO)
        spec_tensor = (spec_tensor - 0.5) / 0.5

        # AUGMENTATION (Noise)
        if self.use_augmentation and self.is_train and self.add_noise_std > 0:
            noise = torch.randn_like(spec_tensor) * self.add_noise_std
            spec_tensor = spec_tensor + noise

        # Label extraction
        filename = os.path.basename(file_path)
        emotion_code = filename.split('_')[2]
        label = EMOTIONS[emotion_code]

        return spec_tensor, torch.tensor(label, dtype=torch.long)


# ==========================================
# BUILD DATALOADERS
# ==========================================
def build_dataloaders(
    spec_path,
    batch_size=32,
    use_augmentation=False,

    # Augmentation params
    add_noise_std=0.005,
    freq_mask_param=10,
    time_mask_param=20,

    seed=42
):
    from sklearn.model_selection import train_test_split
    import os
    from torch.utils.data import DataLoader

    # Load files
    all_files = [
        os.path.join(spec_path, f)
        for f in os.listdir(spec_path)
        if f.endswith(".npy")
    ]

    # Actor-based split
    unique_actors = sorted(list(set([
        os.path.basename(f).split('_')[0] for f in all_files
    ])))

    train_actors, temp_actors = train_test_split(
        unique_actors, test_size=0.2, random_state=seed
    )
    val_actors, test_actors = train_test_split(
        temp_actors, test_size=0.5, random_state=seed
    )

    train_files = [f for f in all_files if os.path.basename(f).split('_')[0] in train_actors]
    val_files   = [f for f in all_files if os.path.basename(f).split('_')[0] in val_actors]
    test_files  = [f for f in all_files if os.path.basename(f).split('_')[0] in test_actors]

    # DATASETS
    train_dataset = CremaDataset(
        train_files,
        is_train=True,
        use_augmentation=use_augmentation,
        add_noise_std=add_noise_std,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param
    )

    val_dataset = CremaDataset(
        val_files,
        is_train=False,
        use_augmentation=use_augmentation,  # safe: won't apply because is_train=False
        add_noise_std=add_noise_std,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param
    )

    test_dataset = CremaDataset(
        test_files,
        is_train=False,
        use_augmentation=use_augmentation,
        add_noise_std=add_noise_std,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param
    )

    # LOADERS
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ==========================================
# BUILD MODEL
# ==========================================
def build_model(
    model_name="resnet18",
    pretrained=False,
    num_classes=6,
    in_channels=1,
    lr=1e-4,
    weight_decay=0.0,
    use_regularization=False,
    dropout_rate=0.5,
    label_smoothing=0.0,
    use_scheduler=False,
    scheduler_factor=0.5,
    scheduler_patience=2,
    device="cuda",
    dinov2_freeze_backbone="freeze_all" # options: "freeze_all", "freeze_except_last", "freeze_except_last_two", "freeze_except_last_three", "finetune_all"
):
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models

    # -------- MODEL --------
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        model.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        num_ftrs = model.fc.in_features

        if use_regularization:
            model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_ftrs, num_classes)
            )
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)

        model.features[0][0] = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)
        num_ftrs = model.classifier[2].in_features

        if use_regularization:
            model.classifier[2] = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_ftrs, num_classes)
            )
        else:
            model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "dinov2" or model_name == "dinov2_small":
        # Load pretrained DINOv2 model
        if model_name == "dinov2":
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=pretrained)
        else:
            print("Loading DINOv2 Small...")
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=pretrained)

        if dinov2_freeze_backbone == "freeze_all":
            # Freeze entire backbone
            for param in model.parameters():
                param.requires_grad = False
        elif dinov2_freeze_backbone == "freeze_except_last":
            # Freeze all but last transformer block + head
            for name, param in model.named_parameters():
                if "blocks.11" in name:  # unfreeze last transformer block
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif dinov2_freeze_backbone == "freeze_except_last_two":
            # Freeze all but last 2 transformer blocks + head
            for name, param in model.named_parameters():
                if "blocks.10" in name or "blocks.11" in name:  # unfreeze last 2 transformer blocks
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif dinov2_freeze_backbone == "freeze_except_last_three":
            # Freeze all but last 3 transformer blocks + head
            for name, param in model.named_parameters():
                if "blocks.9" in name or "blocks.10" in name or "blocks.11" in name:  # unfreeze last 3 transformer blocks
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif dinov2_freeze_backbone == "finetune_all":
            # Finetune entire model (no freezing)
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid dinov2_freeze_backbone option")

        # Get embedding dimension
        embed_dim = model.embed_dim

        # Replace with custom classifier
        if use_regularization:
            head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(embed_dim, num_classes)
            )
        else:
            head = nn.Linear(embed_dim, num_classes)

        # Wrap model
        class DinoClassifier(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x):
                # Convert 1-channel → 3-channel
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)

                features = self.backbone(x)  # shape: (B, embed_dim)
                out = self.head(features)
                return out
            
        model = DinoClassifier(model, head)

    else:
        raise ValueError("Unsupported model")

    model = model.to(device)

    # -------- LOSS --------
    if use_regularization:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # -------- OPTIMIZER --------
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay if use_regularization else 0.0
    )

    # -------- SCHEDULER --------
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=scheduler_factor,
            patience=scheduler_patience
        )

    return model, criterion, optimizer, scheduler


# ==========================================
# TRAIN MODEL
# ==========================================
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    epochs=10,
    device="cuda"
):
    history = {
        "train_acc": [],
        "val_acc": [],
        "lr": []
    }

    for epoch in range(epochs):
        # ---- TRAIN ----
        model.train()
        correct, total = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

        train_acc = 100 * correct / total

        # ---- VALIDATION ----
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)

                _, pred = out.max(1)
                total += y.size(0)
                correct += pred.eq(y).sum().item()

        val_acc = 100 * correct / total

        # ---- SCHEDULER ----
        if scheduler:
            scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | LR: {current_lr:.6f}")

    return history


# ==========================================
# PLOT TRAINING HISTORY
# ==========================================
def plot_history(history, title="Training vs Validation"):
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_acc"]) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(epochs, history["train_acc"], label="Train Acc")
    ax1.plot(epochs, history["val_acc"], label="Val Acc")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left")

    # Optional: Plot learning rate on secondary y-axis
    if "lr" in history and len(history["lr"]) > 0 and len(set(history["lr"])) > 1:
        print(history["lr"])
        ax2 = ax1.twinx()
        ax2.step(epochs, history["lr"], linestyle="--")
        ax2.set_ylabel("Learning Rate")
        ax2.set_yscale("log")

    plt.title(title)
    plt.grid(True)
    plt.show()


# ==========================================
# EVALUATE MODEL
# ==========================================
def evaluate_model(model, test_loader, device="cuda"):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

    acc = 100 * correct / total
    print(f"\nFinal Test Accuracy: {acc:.2f}%")

    return acc

# ==========================================
# VISUALIZE AUGMENTATION
# ==========================================
def visualize_augmentation(file_path, noise_std=0.005, freq_mask=10, time_mask=20):
    """
    Loads a single spectrogram and displays original vs augmented versions.
    """
    # Load and clean
    spec = np.load(file_path)
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    
    orig_spec = spec.copy()
    aug_spec = spec.copy()
    
    n_mels, n_steps = aug_spec.shape
    
    # Frequency and time masking
    if freq_mask > 0 and n_mels > freq_mask:
        f_mask = np.random.randint(1, freq_mask)
        f0 = np.random.randint(0, n_mels - f_mask)
        aug_spec[f0:f0 + f_mask, :] = 0
        
    if time_mask > 0 and n_steps > time_mask:
        t_mask = np.random.randint(1, time_mask)
        t0 = np.random.randint(0, n_steps - t_mask)
        aug_spec[:, t0:t0 + t_mask] = 0

    # Add noise
    aug_tensor = torch.tensor(aug_spec, dtype=torch.float32).unsqueeze(0)
    if noise_std > 0:
        noise = torch.randn_like(aug_tensor) * noise_std
        aug_tensor = aug_tensor + noise
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    img1 = axes[0].imshow(orig_spec, aspect='auto', origin='lower')
    axes[0].set_title("Original Spectrogram")
    fig.colorbar(img1, ax=axes[0])
    
    img2 = axes[1].imshow(aug_tensor.squeeze().numpy(), aspect='auto', origin='lower')
    axes[1].set_title(f"Augmented (Noise={noise_std}, F={freq_mask}, T={time_mask})")
    fig.colorbar(img2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()