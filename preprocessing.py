"""
Prétraitement des images FER2013


Structure attendue :
    fer2013/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── test/
        ├── angry/
        └── ...

"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import cv2

# ─────────────────────────────────────────────────────────────
# 1. CONSTANTES ET CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Ordre alphabétique = ordre qu'ImageFolder assigne automatiquement
# angry=0, disgust=1, fear=2, happy=3, neutral=4, sad=5, surprise=6
EMOTION_LABELS = {
    0: "Colère",
    1: "Dégoût",
    2: "Peur",
    3: "Joie",
    4: "Neutre",
    5: "Tristesse",
    6: "Surprise",
}

IMG_SIZE = 48
MEAN = 0.5071   # Moyenne calculée sur FER2013 (niveaux de gris)
STD  = 0.2520   # Écart-type calculé sur FER2013

# Seuils de bruit pour le débruitage adaptatif (écart-type des résidus)
# Calibrés empiriquement sur FER2013 (images 48×48, niveaux de gris)
NOISE_LOW  = 50    # En dessous : image propre, aucun filtrage
NOISE_HIGH = 150   # Au-dessus  : bruit élevé → Non-Local Means


# ─────────────────────────────────────────────────────────────
# 2. PIPELINES DE TRANSFORMATION
# ─────────────────────────────────────────────────────────────

def _estimate_noise(img_np: np.ndarray) -> float:
    """
    Estime le niveau de bruit via l'écart-type des résidus haute-fréquence.

    Principe : on soustrait une version légèrement lissée à l'image originale.
    Ce qui reste est principalement du bruit — son écart-type est une mesure
    fiable, contrairement à la variance du Laplacien qui mesure la netteté
    globale (une image nette et sans bruit aurait un score élevé au Laplacien).

    Paramètres
    ----------
    img_np : tableau numpy uint8 (H, W), valeurs dans [0, 255]

    Retourne
    --------
    Écart-type des résidus (float). Plus le score est élevé, plus l'image
    est bruitée.
    """
    smoothed = cv2.GaussianBlur(img_np, (5, 5), 1.0)
    residual = img_np.astype(np.float32) - smoothed.astype(np.float32)
    return float(np.std(residual))


class AdaptiveDenoiser(object):
    """
    Débruitage adaptatif intégré dans un pipeline torchvision.
      - Bruit faible  (σ < NOISE_LOW)  → aucun filtrage
      - Bruit modéré  (NOISE_LOW ≤ σ < NOISE_HIGH) → filtre bilatéral léger
        (préserve les contours : meilleur que Gaussien pour les visages)
      - Bruit élevé   (σ ≥ NOISE_HIGH) → Non-Local Means (meilleur pour les
        textures fines mais plus lent)

    Note pickle-safe : aucun objet C++ n'est stocké dans __init__.
    """
    def __init__(self,
                 noise_low: float  = NOISE_LOW,
                 noise_high: float = NOISE_HIGH):
        self.noise_low  = noise_low
        self.noise_high = noise_high

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img, dtype=np.uint8)
        noise  = _estimate_noise(img_np)

        if noise < self.noise_low:
            # Image propre : pas de filtrage
            denoised = img_np

        elif noise < self.noise_high:
            # Bruit modéré : filtre bilatéral (préserve les bords)
            denoised = cv2.bilateralFilter(
                img_np,
                d=5,           # Diamètre du voisinage (réduit pour vitesse)
                sigmaColor=50, # Sensibilité couleur
                sigmaSpace=50, # Sensibilité spatiale
            )
        else:
            # Bruit élevé : Non-Local Means (meilleure qualité)
            denoised = cv2.fastNlMeansDenoising(
                img_np,
                h=10,                  # Force du débruitage
                templateWindowSize=7,
                searchWindowSize=21,
            )

        return Image.fromarray(denoised)


class CLAHEEnhancer(object):
    """
    Égalisation d'histogramme adaptative (CLAHE).

    Appliqué APRÈS le débruitage pour éviter d'amplifier le bruit,
    et SANS sharpening post-CLAHE qui causerait des artefacts de sonnerie
    sur des images 48×48.

    Note pickle-safe : cv2.createCLAHE() est instancié dans __call__.
    """
    def __init__(self, clip_limit: float = 1.5, grid_size: tuple = (4, 4)):
        self.clip_limit = clip_limit
        self.grid_size  = grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np    = np.array(img, dtype=np.uint8)
        # Instancié localement : non-picklable si stocké dans __init__
        clahe     = cv2.createCLAHE(clipLimit=self.clip_limit,
                                     tileGridSize=self.grid_size)
        img_final = clahe.apply(img_np)
        return Image.fromarray(img_final)


class _ForceNLMDenoiser(object):
    """
    Débruitage NLM forcé (indépendant du niveau de bruit estimé).
    Utilisé par le pipeline 'aggressive'.
    Note pickle-safe : pas d'objet C++ stocké dans __init__.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        img_np   = np.array(img, dtype=np.uint8)
        denoised = cv2.fastNlMeansDenoising(
            img_np, h=15, templateWindowSize=7, searchWindowSize=21
        )
        return Image.fromarray(denoised)


def _build_preprocessing_steps(pipeline: str) -> list:
    """
    Retourne la liste des transforms de prétraitement selon le pipeline choisi.

    "minimal"    → [CLAHEEnhancer]
    "light"      → [AdaptiveDenoiser, CLAHEEnhancer]   ← recommandé
    "aggressive" → [_ForceNLMDenoiser, CLAHEEnhancer]
    """
    if pipeline == "minimal":
        return [CLAHEEnhancer()]
    elif pipeline == "light":
        return [AdaptiveDenoiser(), CLAHEEnhancer()]
    elif pipeline == "aggressive":
        return [_ForceNLMDenoiser(), CLAHEEnhancer()]
    else:
        raise ValueError(
            f"Pipeline inconnu : '{pipeline}'. "
            "Choisissez parmi 'minimal', 'light', 'aggressive'."
        )


def _make_train_transform(pipeline: str = "light") -> transforms.Compose:
    """
    Construit le pipeline d'entraînement (avec augmentations).

    Paramètres
    ----------
    pipeline : "minimal" | "light" | "aggressive"
        Détermine l'intensité du débruitage avant CLAHE.
    """
    preprocessing_steps = _build_preprocessing_steps(pipeline)

    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        *preprocessing_steps,
        # ── Augmentations géométriques ──────────────────────────────────────
        transforms.RandomHorizontalFlip(p=0.5),
        # Les émotions sont symétriques gauche/droite → double le dataset effectif

        transforms.RandomRotation(degrees=10),
        # Compense les légères inclinaisons de tête

        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        # Simule de petits décalages de cadrage caméra

        # ── Augmentations photométriques ────────────────────────────────────
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        # FER2013 présente des variations d'illumination importantes

        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        # Robustesse aux images légèrement floues

        # ── Conversion et normalisation ─────────────────────────────────────
        transforms.ToTensor(),
        transforms.Normalize(mean=[MEAN], std=[STD]),

        # ── Régularisation par masquage ─────────────────────────────────────
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
        # Simule des occultations partielles (lunettes, mains, masques)
    ])


def _make_val_transform(pipeline: str = "light") -> transforms.Compose:
    """
    Construit le pipeline de validation / test (déterministe, sans augmentation).

    Paramètres
    ----------
    pipeline : "minimal" | "light" | "aggressive"
    """
    preprocessing_steps = _build_preprocessing_steps(pipeline)

    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        *preprocessing_steps,
        transforms.ToTensor(),
        transforms.Normalize(mean=[MEAN], std=[STD]),
    ])


# ── Instances par défaut (pipeline "light") ────────────────────────────────────
train_transform = _make_train_transform(pipeline="light")
val_transform   = _make_val_transform(pipeline="light")


# ─────────────────────────────────────────────────────────────
# 3. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────

class _TransformSubset(torch.utils.data.Dataset):
    """
    Sous-ensemble d'un ImageFolder avec un transform indépendant.

    Pourquoi : utiliser deux instances séparées d'ImageFolder pour train et val
    est fragile — si l'ordre des fichiers diffère entre les deux instances,
    les indices ne correspondent plus et on risque une fuite de données.
    Cette classe garantit que train et val partagent exactement les mêmes
    métadonnées (samples, targets) tout en appliquant des transforms distincts.
    """
    def __init__(self, base_dataset, indices, transform):
        self.base      = base_dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.base.samples[self.indices[idx]]
        img = self.base.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(
    data_root:   str,
    batch_size:  int   = 64,
    val_split:   float = 0.2,
    num_workers: int   = 2,
    pipeline:    str   = "light",
):
    """
    Charge le dataset FER2013 depuis un dossier d'images.

    Paramètres
    ----------
    data_root   : chemin vers le dossier contenant 'train/' et 'test/'
    batch_size  : taille des mini-batches
    val_split   : proportion du train utilisée pour la validation (défaut 20 %)
    num_workers : processus parallèles pour le chargement
    pipeline    : "minimal" | "light" | "aggressive"

    Retourne
    --------
    train_loader, val_loader, test_loader

    Garanties
    ---------
    - Aucune image de test ne figure dans train ou val  [FIX 1]
    - Le split train/val est reproductible (seed=42)
    - Les transforms sont appliqués à la volée           [FIX 2]
    - persistent_workers=True évite de respawner les workers entre époques [FIX 6]
    """
    t_transform = _make_train_transform(pipeline)
    v_transform = _make_val_transform(pipeline)

    # Une seule instance — garantit que train et val partagent les mêmes métadonnées
    full_ds    = datasets.ImageFolder(root=os.path.join(data_root, "train"))
    total      = len(full_ds)
    val_size   = int(total * val_split)
    train_size = total - val_size

    # Indices fixes (reproductible)
    train_indices, val_indices = random_split(
        range(total), [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_subset = _TransformSubset(full_ds, list(train_indices), t_transform)
    val_subset   = _TransformSubset(full_ds, list(val_indices),   v_transform)

    # Test : instance séparée — jamais mélangée avec le train
    test_ds = datasets.ImageFolder(
        root=os.path.join(data_root, "test"),
        transform=v_transform,
    )

    use_persistent = num_workers > 0

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=use_persistent,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=use_persistent,
    )

    print(f"  Pipeline    : {pipeline}")
    print(f"  Train       : {len(train_subset):>6} images")
    print(f"  Val         : {len(val_subset):>6} images")
    print(f"  Test        : {len(test_ds):>6} images")

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────
# 4. CALCUL DES POIDS DE CLASSES
# ─────────────────────────────────────────────────────────────

def compute_class_weights(data_root: str) -> torch.Tensor:
    """
    Calcule des poids inversement proportionnels à la fréquence de chaque classe.
    À passer à nn.CrossEntropyLoss(weight=...) pour gérer le déséquilibre.
    """
    train_ds  = datasets.ImageFolder(root=os.path.join(data_root, "train"))
    counts    = Counter([label for _, label in train_ds.samples])
    total     = sum(counts.values())
    n_classes = len(counts)

    weights = torch.zeros(n_classes)
    for cls in range(n_classes):
        weights[cls] = total / (n_classes * counts[cls])

    weights = weights / weights.sum() * n_classes
    return weights


# ─────────────────────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────────────────────

def plot_class_distribution(data_root: str, save_path: str = None):
    """Graphique de la distribution des classes dans le dossier train."""
    train_ds = datasets.ImageFolder(root=os.path.join(data_root, "train"))
    counts   = Counter([label for _, label in train_ds.samples])

    labels = [EMOTION_LABELS[i] for i in range(len(counts))]
    values = [counts[i] for i in range(len(counts))]
    colors = ["#E74C3C", "#8E44AD", "#2ECC71", "#F39C12", "#95A5A6", "#3498DB", "#1ABC9C"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Distribution des classes – FER2013 (Train)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Nombre d'images")
    ax.set_xlabel("Émotion")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                str(val), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Sauvegardé : {save_path}")
    plt.show()


def plot_sample_grid(data_root: str, n_per_class: int = 4, save_path: str = None):
    """Grille d'exemples bruts (avant prétraitement) pour chaque émotion."""
    ds = datasets.ImageFolder(
        root=os.path.join(data_root, "train"),
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
        ]),
    )

    targets = np.array(ds.targets)
    indices_by_class = {
        c: np.where(targets == c)[0][:n_per_class].tolist()
        for c in range(len(EMOTION_LABELS))
    }
    samples_by_class = {
        c: [np.array(ds[i][0]).squeeze() for i in idxs]
        for c, idxs in indices_by_class.items()
    }

    n_classes = len(EMOTION_LABELS)
    fig, axes = plt.subplots(n_classes, n_per_class,
                              figsize=(n_per_class * 2, n_classes * 2.2))
    for row in range(n_classes):
        for col in range(n_per_class):
            axes[row, col].imshow(samples_by_class[row][col], cmap="gray")
            if col == 0:
                axes[row, col].set_ylabel(EMOTION_LABELS[row], fontsize=9,
                                          rotation=0, labelpad=55, va="center")
            axes[row, col].axis("off")

    fig.suptitle("Exemples par émotion – FER2013 (images brutes)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Sauvegardé : {save_path}")
    plt.show()


def plot_pipeline_comparison(data_root: str, save_path: str = None):
    """
    Compare les trois pipelines (minimal / light / aggressive) côte à côte
    sur une image par émotion.

    Remplace plot_preprocessed_samples de la version originale :
    - Chaque colonne prétraitée montre un pipeline différent (résultats
      visuellement distincts, contrairement à l'ancienne version où les
      colonnes étaient identiques car val_transform est déterministe).
    - Le score de bruit estimé est affiché pour chaque variante.
    """
    pipeline_names = ["minimal", "light", "aggressive"]

    # Dataset brut (référence)
    ds_raw = datasets.ImageFolder(
        root=os.path.join(data_root, "train"),
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
        ]),
    )

    # Un dataset par pipeline
    ds_by_pipeline = {
        name: datasets.ImageFolder(
            root=os.path.join(data_root, "train"),
            transform=_make_val_transform(pipeline=name),
        )
        for name in pipeline_names
    }

    targets = np.array(ds_raw.targets)
    class_indices = {
        c: int(np.where(targets == c)[0][0])
        for c in range(len(EMOTION_LABELS))
    }

    n_classes = len(EMOTION_LABELS)
    n_cols    = 1 + len(pipeline_names)
    fig, axes = plt.subplots(n_classes, n_cols,
                              figsize=(n_cols * 2.4, n_classes * 2.6))

    col_titles = ["Brut\n(référence)"] + [f"Pipeline\n{n}" for n in pipeline_names]

    for row, cls in enumerate(range(n_classes)):
        idx     = class_indices[cls]
        raw_np  = np.array(ds_raw[idx][0]).squeeze()

        # Colonne 0 : image brute
        noise_raw = _estimate_noise(raw_np)
        axes[row, 0].imshow(raw_np, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_ylabel(EMOTION_LABELS[cls], fontsize=9,
                                rotation=0, labelpad=55, va="center")
        axes[row, 0].set_xlabel(f"bruit={noise_raw:.1f}", fontsize=7)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title(col_titles[0], fontsize=8)

        # Colonnes 1-3 : sorties des trois pipelines
        for col, name in enumerate(pipeline_names, start=1):
            tensor, _  = ds_by_pipeline[name][idx]
            display    = (tensor.squeeze().numpy() * STD + MEAN).clip(0, 1)
            display_u8 = (display * 255).astype(np.uint8)
            noise_pp   = _estimate_noise(display_u8)

            axes[row, col].imshow(display, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_xlabel(f"bruit={noise_pp:.1f}", fontsize=7)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=8)

    fig.suptitle(
        "Comparaison des pipelines de prétraitement\n"
        "Grayscale → Resize → [Denoiser] → CLAHE → Normalize",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Sauvegardé : {save_path}")
    plt.show()


def plot_augmentation_comparison(data_root: str, n_augmented: int = 5,
                                 save_path: str = None):
    """Montre une image originale et plusieurs versions augmentées côte à côte."""
    ds_raw = datasets.ImageFolder(
        root=os.path.join(data_root, "train"),
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
        ]),
    )
    raw_img, label = ds_raw[0]
    raw_np = np.array(raw_img).squeeze()

    aug_only = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
    ])

    pil_img = Image.fromarray(raw_np, mode="L")

    fig, axes = plt.subplots(1, n_augmented + 1, figsize=(3 * (n_augmented + 1), 3.2))
    axes[0].imshow(raw_np, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Original\n({EMOTION_LABELS[label]})", fontsize=9)
    axes[0].axis("off")

    for i in range(n_augmented):
        aug_img = aug_only(pil_img)
        axes[i + 1].imshow(np.array(aug_img).squeeze(), cmap="gray", vmin=0, vmax=255)
        axes[i + 1].set_title(f"Aug #{i+1}", fontsize=9)
        axes[i + 1].axis("off")

    fig.suptitle("Avant / Après augmentation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Sauvegardé : {save_path}")
    plt.show()


def plot_normalization_effect(data_root: str, save_path: str = None):
    """Illustre l'effet de la normalisation Z-score sur un exemple."""
    ds_raw = datasets.ImageFolder(
        root=os.path.join(data_root, "train"),
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
        ]),
    )
    raw_img, _ = ds_raw[0]
    raw_np = np.array(raw_img).squeeze()

    tensor_raw  = transforms.ToTensor()(raw_img)
    tensor_norm = transforms.Normalize([MEAN], [STD])(tensor_raw.clone())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    axes[0].imshow(raw_np, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Image brute (uint8)\nPlage : [0, 255]", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(tensor_raw.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        f"Après ToTensor()\nPlage : [0, 1]  |  Moy={tensor_raw.mean():.3f}", fontsize=10
    )
    axes[1].axis("off")

    norm_np = tensor_norm.squeeze().numpy()
    axes[2].imshow(norm_np, cmap="gray")
    axes[2].set_title(
        f"Après Normalize(μ={MEAN}, σ={STD})\n"
        f"Moy≈{norm_np.mean():.3f}  Std≈{norm_np.std():.3f}",
        fontsize=10,
    )
    axes[2].axis("off")

    fig.suptitle("Effet de la normalisation Z-score", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Sauvegardé : {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# 6. COMPARAISON ORIGINAL VS PRÉTRAITÉ (PIXELS NETS)
# ─────────────────────────────────────────────────────────────

def compare_original_vs_preprocessed(
    data_root:   str,
    n_images:    int = 7,
    px_per_cell: int = 288,
    pipeline:    str = "light",
    save_path:   str = None,
):
    """
    Affiche les images originales et prétraitées côte à côte avec des pixels nets.

    Pourquoi les images semblent floues quand on ouvre le PNG sur PC :
      imshow(..., interpolation="nearest") empeche matplotlib de flouter
      lors du rendu dans la figure, mais si la figure sauvegardee est petite
      (ex. 48 px par cellule), le visualiseur Windows/Mac doit l'agrandir
      lui-meme et applique son propre lissage.

    Solution : on agrandit les images en RAM avec nearest-neighbor AVANT
    de les passer a imshow, puis on fixe figsize pour que chaque cellule
    fasse exactement px_per_cell pixels dans le PNG final. Ainsi le
    visualiseur n'a plus besoin d'agrandir quoi que ce soit.

    Parametres
    ----------
    data_root   : chemin vers le dossier contenant 'train/'
    n_images    : nombre de colonnes (une emotion par colonne, max 7)
    px_per_cell : taille en pixels de chaque cellule dans le PNG sauvegarde
                  (288 -> affichage net sur ecrans Full HD et plus)
    pipeline    : "minimal" | "light" | "aggressive"
    save_path   : chemin de sauvegarde optionnel
    """
    ds_raw = datasets.ImageFolder(
        root=os.path.join(data_root, "train"),
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
        ]),
    )
    ds_preprocessed = datasets.ImageFolder(
        root=os.path.join(data_root, "train"),
        transform=_make_val_transform(pipeline=pipeline),
    )

    targets = np.array(ds_raw.targets)
    n_cols  = min(n_images, len(EMOTION_LABELS))
    indices = [
        int(np.where(targets == c)[0][0])
        for c in range(n_cols)
    ]

    # Zoom en RAM avec nearest-neighbor : chaque pixel devient un bloc de
    # zoom_factor x zoom_factor pixels identiques -> aucun lissage introduit
    zoom_factor = px_per_cell // IMG_SIZE  # ex. 288 // 48 = 6

    def _zoom_nearest(arr_2d):
        return arr_2d.repeat(zoom_factor, axis=0).repeat(zoom_factor, axis=1)

    dpi       = 100
    cell_inch = px_per_cell / dpi
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(n_cols * cell_inch, 2 * cell_inch + 0.6),
    )
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, idx in enumerate(indices):
        emotion = EMOTION_LABELS[ds_raw.targets[idx]]

        # Ligne 0 : original
        raw_np  = np.array(ds_raw[idx][0]).squeeze()
        raw_big = _zoom_nearest(raw_np)
        axes[0, col].imshow(raw_big, cmap="gray", vmin=0, vmax=255,
                            interpolation="nearest")
        axes[0, col].set_title(emotion, fontsize=8)
        axes[0, col].axis("off")

        # Ligne 1 : pretraite
        tensor, _ = ds_preprocessed[idx]
        display   = (tensor.squeeze().numpy() * STD + MEAN).clip(0, 1)
        disp_u8   = (display * 255).astype(np.uint8)
        prep_big  = _zoom_nearest(disp_u8)
        axes[1, col].imshow(prep_big, cmap="gray", vmin=0, vmax=255,
                            interpolation="nearest")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original",  fontsize=9, labelpad=4)
    axes[1, 0].set_ylabel("Pretraite", fontsize=9, labelpad=4)

    fig.suptitle(
        f"Original vs Pretraite - pipeline '{pipeline}'  "
        f"(zoom x{zoom_factor}, nearest-neighbor)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi)
        print(f"  Sauvegarde : {save_path}  "
              f"({n_cols * px_per_cell} x {2 * px_per_cell} px)")
    plt.show()


# ─────────────────────────────────────────────────────────────
# 7. SAUVEGARDE D'ÉCHANTILLONS PRÉTRAITÉS
# ─────────────────────────────────────────────────────────────

def save_preprocessed_samples(
    data_root:   str,
    output_dir:  str = "preprocessed_samples",
    n_per_class: int = 5,
    pipeline:    str = "light",
):
    """
    Sauvegarde des images prétraitées sur disque, organisées par émotion.

    Structure créée :
        output_dir/
        ├── angry/
        │   ├── sample_00.png
        │   └── ...
        └── ...

    Paramètres
    ----------
    data_root    : chemin vers le dossier contenant 'train/'
    output_dir   : dossier de sortie (créé s'il n'existe pas)
    n_per_class  : nombre d'images sauvegardées par émotion
    pipeline     : "minimal" | "light" | "aggressive"
    """
    v_transform = _make_val_transform(pipeline)
    ds = datasets.ImageFolder(
        root=os.path.join(data_root, "train"),
        transform=v_transform,
    )

    targets          = np.array(ds.targets)
    indices_by_class = {
        c: np.where(targets == c)[0][:n_per_class].tolist()
        for c in range(len(ds.classes))
    }

    saved_count = 0
    for cls_idx, cls_name in enumerate(ds.classes):
        cls_dir = os.path.join(output_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        for sample_num, ds_idx in enumerate(indices_by_class[cls_idx]):
            tensor, _ = ds[ds_idx]
            display   = (tensor.squeeze().numpy() * STD + MEAN).clip(0, 1)
            img_uint8 = (display * 255).astype(np.uint8)
            Image.fromarray(img_uint8, mode="L").save(
                os.path.join(cls_dir, f"sample_{sample_num:02d}.png")
            )
            saved_count += 1

    print(f"\n  Échantillons sauvegardés  ({pipeline})")
    print(f"  Dossier  : {os.path.abspath(output_dir)}")
    print(f"  Fichiers : {saved_count} images "
          f"({n_per_class} par classe × {len(ds.classes)} classes)")


# ─────────────────────────────────────────────────────────────
# 8. VÉRIFICATION DU PIPELINE
# ─────────────────────────────────────────────────────────────

def verify_pipeline(data_root: str, pipeline: str = "light"):
    """Affiche les statistiques d'un batch et les poids de classes."""
    print("\n" + "=" * 55)
    print(f"  Vérification du pipeline : {pipeline}")
    print("=" * 55)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root, batch_size=64, pipeline=pipeline
    )

    imgs, labels = next(iter(train_loader))
    print(f"\n  Forme du batch  : {imgs.shape}")
    print(f"  Type            : {imgs.dtype}")
    print(f"  Min / Max       : {imgs.min():.4f} / {imgs.max():.4f}")
    print(f"  Moyenne         : {imgs.mean():.4f}  (≈ 0 attendu)")
    print(f"  Écart-type      : {imgs.std():.4f}  (≈ 1 attendu)")
    print(f"  Labels présents : {sorted(labels.unique().tolist())}")

    weights = compute_class_weights(data_root)
    print("\n  Poids de classes (pour CrossEntropyLoss) :")
    for i, w in enumerate(weights):
        print(f"    {EMOTION_LABELS[i]:<12} : {w:.4f}")

    print("=" * 55 + "\n")
    return train_loader, val_loader, test_loader



# ─────────────────────────────────────────────────────────────
# 9. POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Modifiez ce chemin selon votre organisation ──────────
    DATA_ROOT = "C:/Users/Leith/.cache/kagglehub/datasets/msambare/fer2013/versions/1/"
    # ── Choisissez le pipeline : "minimal" | "light" | "aggressive"
    PIPELINE  = "light"
    # ─────────────────────────────────────────────────────────

    if not os.path.isdir(os.path.join(DATA_ROOT, "train")):
        print(f"[ERREUR] Dossier '{DATA_ROOT}train/' introuvable.")
        print("         Vérifiez le chemin DATA_ROOT et la structure du dataset.")
        exit(1)

    # Vérification complète du pipeline
    train_loader, val_loader, test_loader = verify_pipeline(DATA_ROOT, pipeline=PIPELINE)

    # Visualisations
    print("[0/7] Sauvegarde des échantillons prétraités...")
    save_preprocessed_samples(
        DATA_ROOT,
        output_dir="preprocessed_samples",
        n_per_class=5,
        pipeline=PIPELINE,
    )

    print("[1/7] Distribution des classes...")
    plot_class_distribution(DATA_ROOT, save_path="distribution_classes.png")

    print("[2/7] Grille d'exemples par émotion...")
    plot_sample_grid(DATA_ROOT, n_per_class=4, save_path="exemples_emotions.png")

    print("[3/7] Comparaison des trois pipelines...")
    plot_pipeline_comparison(DATA_ROOT, save_path="comparaison_pipelines.png")

    print("[4/7] Comparaison avant/après augmentation...")
    plot_augmentation_comparison(DATA_ROOT, n_augmented=5, save_path="augmentation.png")

    print("[5/7] Effet de la normalisation...")
    plot_normalization_effect(DATA_ROOT, save_path="normalisation.png")

    print("[6/7] Original vs prétraité (pixels nets)...")
    compare_original_vs_preprocessed(
        DATA_ROOT,
        n_images=7,           # une colonne par émotion
        px_per_cell=288,      # 288px par cellule → net sur tout écran
        pipeline=PIPELINE,
        save_path="original_vs_pretraite.png",
    )

    print("\n  Prétraitement terminé. Prêt pour l'entraînement.")