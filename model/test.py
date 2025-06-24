from scipy.ndimage import distance_transform_edt, binary_erosion
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import os

# PyTorch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset, random_split

from DMBMSNet import DMBMSNet
from DresdenDataset import DresdenDataset
from HemoSetDataset import HemoSetDataset
from CholecSeg8kDataset import CholecSeg8kDataset

torch.manual_seed(1)
cudnn.benchmark = True
warnings.simplefilter("ignore", category=FutureWarning)


# ------------------------- #
# Fonctions de calcul des métriques globales
# ------------------------- #

def accuracy_score(pred, gt):
    """ Calcule l'accuracy pixel par pixel. """
    return np.sum(pred == gt) / pred.size

def dice_score_per_sample(pred, gt, num_classes):
    eps = 1e-6
    scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.float32)
        gt_c = (gt == c).astype(np.float32)
        intersection = np.sum(pred_c * gt_c)
        denominator = np.sum(pred_c) + np.sum(gt_c)
        if denominator == 0:
            dice = 1.0
        else:
            dice = (2. * intersection + eps) / (denominator + eps)
        scores.append(dice)
    return np.mean(scores)

def jaccard_score_per_sample(pred, gt, num_classes):
    eps = 1e-6
    scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.float32)
        gt_c = (gt == c).astype(np.float32)
        intersection = np.sum(pred_c * gt_c)
        union = np.sum(pred_c) + np.sum(gt_c) - intersection
        if union == 0:
            score = 1.0
        else:
            score = (intersection + eps) / (union + eps)
        scores.append(score)
    return np.mean(scores)

def compute_hd95(binary1, binary2):
    """ Calcule la Hausdorff Distance 95 pour deux masques binaires. """
    struct = np.ones((3, 3), dtype=bool)
    edge1 = binary1 ^ binary_erosion(binary1, structure=struct)
    edge2 = binary2 ^ binary_erosion(binary2, structure=struct)
    if not edge1.any() or not edge2.any():
        return np.nan
    dt_edge2 = distance_transform_edt(~edge2)
    distances1 = dt_edge2[edge1]
    dt_edge1 = distance_transform_edt(~edge1)
    distances2 = dt_edge1[edge2]
    all_distances = np.concatenate([distances1, distances2])
    hd95 = np.percentile(all_distances, 95)
    return hd95

def hausdorff95_score_per_sample(pred, gt, num_classes):
    scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.bool_)
        gt_c = (gt == c).astype(np.bool_)
        if not pred_c.any() and not gt_c.any():
            hd95 = 0.0
        elif not pred_c.any() or not gt_c.any():
            hd95 = np.inf
        else:
            hd95 = compute_hd95(pred_c, gt_c)
        scores.append(hd95)
    finite_scores = [s for s in scores if np.isfinite(s)]
    if len(finite_scores) == 0:
        return np.inf
    return np.mean(finite_scores)


# ------------------------- #
# Fonctions de calcul des métriques par classe
# ------------------------- #

def accuracy_score_per_class(pred, gt, num_classes):
    scores = []
    for c in range(num_classes):
        gt_mask = (gt == c)
        if np.sum(gt_mask) == 0:
            # Si aucune occurrence dans le ground truth, on considère l'accuracy comme parfaite
            acc = 1.0
        else:
            correct = np.sum((pred == c) & gt_mask)
            acc = correct / np.sum(gt_mask)
        scores.append(acc)
    return scores

def dice_score_per_class(pred, gt, num_classes):
    eps = 1e-6
    scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.float32)
        gt_c = (gt == c).astype(np.float32)
        intersection = np.sum(pred_c * gt_c)
        denominator = np.sum(pred_c) + np.sum(gt_c)
        if denominator == 0:
            dice = 1.0
        else:
            dice = (2. * intersection + eps) / (denominator + eps)
        scores.append(dice)
    return scores

def jaccard_score_per_class(pred, gt, num_classes):
    eps = 1e-6
    scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.float32)
        gt_c = (gt == c).astype(np.float32)
        intersection = np.sum(pred_c * gt_c)
        union = np.sum(pred_c) + np.sum(gt_c) - intersection
        if union == 0:
            score = 1.0
        else:
            score = (intersection + eps) / (union + eps)
        scores.append(score)
    return scores

def hausdorff95_score_per_class(pred, gt, num_classes):
    scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.bool_)
        gt_c = (gt == c).astype(np.bool_)
        if not pred_c.any() and not gt_c.any():
            hd95 = 0.0
        elif not pred_c.any() or not gt_c.any():
            hd95 = np.inf
        else:
            hd95 = compute_hd95(pred_c, gt_c)
        scores.append(hd95)
    return scores


# ------------------------- #
# Boucle de test
# ------------------------- #

def test_model(model, test_loader, num_classes=9, device="cuda", save_results=False, results_dir="results"):
    """
    Teste le modèle et calcule :
      - Des métriques globales par image,
      - Des métriques par classe par image (mais uniquement si la classe est présente dans le ground truth),
      - Les moyennes globales et agrégées par classe ne considérant que les images où la classe est présente.
    
    Si save_results=True, les résultats seront sauvegardés dans results_dir sous forme de fichiers CSV.
    """
    os.makedirs(results_dir, exist_ok=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()

    test_loss = 0
    global_metrics_list = []  # pour stocker les métriques globales moyennes par batch (afin de faire une moyenne globale)
    results_per_image = []    # liste de dictionnaires, chacun contenant les métriques pour une image
    # Pour l'agrégation par classe, on utilisera des dictionnaires listant les valeurs uniquement pour les images qui possèdent la classe
    agg_metrics = {
        "accuracy": {c: [] for c in range(num_classes)},
        "dice": {c: [] for c in range(num_classes)},
        "jaccard": {c: [] for c in range(num_classes)},
        "hd95": {c: [] for c in range(num_classes)},
    }

    # Génération dynamique des noms de classes
    class_names = [f"Classe {i+1}" for i in range(num_classes)]

    image_counter = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = [img.squeeze(1).to(device) for img in images]
            masks  = [mask.squeeze(1).to(device) for mask in masks]
            current_frame = images[-1]
            current_mask  = masks[-1]
            memory_frames = images[:-1]
            memory_masks  = masks[:-1]

            prediction = model(current_frame, training=True, memory_frames=memory_frames, memory_masks=memory_masks)
            loss = criterion(prediction, current_mask.argmax(dim=1).long())
            print(f"[{image_counter+1}/{len(test_loader)*len(prediction)}] Testing loss: {loss.item():.4f}")
            test_loss += loss.item()

            # Conversion des masques en arrays numpy
            pred_masks = prediction.argmax(dim=1).detach().cpu().numpy()  # forme: (batch, H, W)
            target_masks = current_mask.argmax(dim=1).detach().cpu().numpy()  # forme: (batch, H, W)

            # Calcul pour chaque image du batch
            for i in range(pred_masks.shape[0]):
                image_counter += 1
                # Calcul des métriques globales pour l'image
                acc = accuracy_score(pred_masks[i], target_masks[i])
                dice = dice_score_per_sample(pred_masks[i], target_masks[i], num_classes)
                jaccard = jaccard_score_per_sample(pred_masks[i], target_masks[i], num_classes)
                hd95 = hausdorff95_score_per_sample(pred_masks[i], target_masks[i], num_classes)

                # Calcul des métriques par classe pour l'image
                acc_pc = accuracy_score_per_class(pred_masks[i], target_masks[i], num_classes)
                dice_pc = dice_score_per_class(pred_masks[i], target_masks[i], num_classes)
                jaccard_pc = jaccard_score_per_class(pred_masks[i], target_masks[i], num_classes)
                hd95_pc = hausdorff95_score_per_class(pred_masks[i], target_masks[i], num_classes)

                # Initialisation d'un dictionnaire pour stocker les résultats de cette image
                image_results = {
                    "Image_Index": image_counter,
                    "Global_Accuracy": acc,
                    "Global_Dice": dice,
                    "Global_Jaccard": jaccard,
                    "Global_Hd95": hd95
                }

                # Pour chaque classe, on enregistre la métrique uniquement si la classe est présente dans le ground truth
                for c in range(num_classes):
                    # Vérifier si la classe c est présente dans le masque ground truth pour cette image
                    if np.sum(target_masks[i] == c) > 0:
                        image_results[f"Accuracy_{class_names[c]}"] = acc_pc[c]
                        image_results[f"Dice_{class_names[c]}"] = dice_pc[c]
                        image_results[f"Jaccard_{class_names[c]}"] = jaccard_pc[c]
                        image_results[f"Hd95_{class_names[c]}"] = hd95_pc[c]
                        # Stockage pour l'agrégation finale par classe
                        agg_metrics["accuracy"][c].append(acc_pc[c])
                        agg_metrics["dice"][c].append(dice_pc[c])
                        agg_metrics["jaccard"][c].append(jaccard_pc[c])
                        agg_metrics["hd95"][c].append(hd95_pc[c])
                    else:
                        # La classe n'est pas présente : on peut noter la valeur comme NaN, ou tout simplement ne rien stocker pour l'image
                        image_results[f"Accuracy_{class_names[c]}"] = np.nan
                        image_results[f"Dice_{class_names[c]}"] = np.nan
                        image_results[f"Jaccard_{class_names[c]}"] = np.nan
                        image_results[f"Hd95_{class_names[c]}"] = np.nan

                results_per_image.append(image_results)

            # Nettoyage
            del current_frame, current_mask, memory_frames, memory_masks, prediction, loss
            torch.cuda.empty_cache()

    avg_test_loss = test_loss / len(test_loader)
    print("------------------------------")
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Calcul des métriques globales moyennes sur toutes les images
    all_global_acc = [d["Global_Accuracy"] for d in results_per_image]
    all_global_dice = [d["Global_Dice"] for d in results_per_image]
    all_global_jaccard = [d["Global_Jaccard"] for d in results_per_image]
    all_global_hd95 = [d["Global_Hd95"] for d in results_per_image if np.isfinite(d["Global_Hd95"])]

    avg_global_acc = np.mean(all_global_acc)
    avg_global_dice = np.mean(all_global_dice)
    avg_global_jaccard = np.mean(all_global_jaccard)
    avg_global_hd95 = np.mean(all_global_hd95) if len(all_global_hd95) > 0 else np.inf

    print(f"Global Accuracy: {avg_global_acc:.4f}")
    print(f"Global Dice Coefficient: {avg_global_dice:.4f}")
    print(f"Global Jaccard Index: {avg_global_jaccard:.4f}")
    print(f"Global Hausdorff Distance 95: {avg_global_hd95:.4f}")
    print("------------------------------")

    # Calcul des moyennes par classe pour les images qui contiennent chacune la classe
    avg_per_class = {}
    for c in range(num_classes):
        if agg_metrics["accuracy"][c]:
            avg_per_class[class_names[c]] = {
                "Accuracy": np.mean(agg_metrics["accuracy"][c]),
                "Dice": np.mean(agg_metrics["dice"][c]),
                "Jaccard": np.mean(agg_metrics["jaccard"][c]),
                "Hd95": np.mean([v for v in agg_metrics["hd95"][c] if np.isfinite(v)]) if any(np.isfinite(agg_metrics["hd95"][c])) else np.inf
            }
        else:
            # Si aucune image ne contient cette classe
            avg_per_class[class_names[c]] = {
                "Accuracy": np.nan,
                "Dice": np.nan,
                "Jaccard": np.nan,
                "Hd95": np.nan,
            }

    print("Métriques moyennes par classe (seulement sur les images possédant la classe) :")
    for cl, metrics in avg_per_class.items():
        print(f"{cl}:")
        print(f"   Accuracy: {metrics['Accuracy']:.4f}")
        print(f"   Dice Coefficient: {metrics['Dice']:.4f}")
        print(f"   Jaccard Index: {metrics['Jaccard']:.4f}")
        print(f"   Hausdorff Distance 95: {metrics['Hd95']:.4f}")

    # Sauvegarde des résultats dans des fichiers CSV si demandé
    if save_results:
        # Résultats par image
        df_images = pd.DataFrame(results_per_image)
        df_images.to_csv(os.path.join(results_dir, "per_image_metrics.csv"), index=False)
        print(f"Résultats par image sauvegardés dans {os.path.join(results_dir, 'per_image_metrics.csv')}")

        # Agrégation globale
        global_results = {
            "Test_Loss": avg_test_loss,
            "Global_Accuracy": avg_global_acc,
            "Global_Dice": avg_global_dice,
            "Global_Jaccard": avg_global_jaccard,
            "Global_Hd95": avg_global_hd95
        }
        # On peut sauvegarder également les métriques moyennes par classe
        df_global = pd.DataFrame(global_results, index=[0])
        df_class = pd.DataFrame(avg_per_class).transpose().reset_index().rename(columns={"index": "Classe"})
        df_global.to_csv(os.path.join(results_dir, "global_metrics.csv"), index=False)
        df_class.to_csv(os.path.join(results_dir, "per_class_metrics.csv"), index=False)
        print(f"Résultats globaux sauvegardés dans {results_dir}")

    return {
        "global": {
            "loss": avg_test_loss,
            "accuracy": avg_global_acc,
            "dice": avg_global_dice,
            "jaccard": avg_global_jaccard,
            "hd95": avg_global_hd95,
        },
        "per_class": avg_per_class,
        "per_image": results_per_image
    }


if __name__ == '__main__':

    # Dresden Dataset

    dataset_1 = DresdenDataset(
        data_dir=r"D:\archive\Dresden Dataset\dresden_tensors",
        history_length=4,
        augment=False
    )

    train_vids_1 = {4, 5, 8, 10, 12, 16, 17, 22, 23, 24, 25, 27, 29, 30, 31}
    val_vids_1   = {2, 7, 11, 18, 20}
    test_vids_1  = {3, 21, 26}

    train_indices_1 = []
    val_indices_1 = []
    test_indices_1 = []

    for idx, seq in enumerate(dataset_1.sequences):
        if isinstance(seq, tuple):
            files = seq[0]
        else:
            files = seq
        video_id = int(os.path.basename(files[0]).split('_')[0])
        if video_id in train_vids_1:
            train_indices_1.append(idx)
        elif video_id in val_vids_1:
            val_indices_1.append(idx)
        elif video_id in test_vids_1:
            test_indices_1.append(idx)

    train_set_1 = Subset(dataset_1, train_indices_1)
    val_set_1 = Subset(dataset_1, val_indices_1)
    test_set_1 = Subset(dataset_1, test_indices_1)

    # HemoSet Dataset

    dataset_2 = HemoSetDataset(
        data_dir=r"D:\HemoSet\hemoset_tensors",
        history_length=4,
        augment=False
    )

    train_vids_2 = {1, 2, 3, 4, 5, 6}
    val_vids_2 = {7, 9}
    test_vids_2 = {10, 11}

    train_indices_2 = []
    val_indices_2 = []
    test_indices_2 = []

    for idx, seq in enumerate(dataset_2.sequences):
        if isinstance(seq, tuple):
            files = seq[0]
        else:
            files = seq
        video_id = int(os.path.basename(files[0]).split('_')[0])
        if video_id in train_vids_2:
            train_indices_2.append(idx)
        elif video_id in val_vids_2:
            val_indices_2.append(idx)
        elif video_id in test_vids_2:
            test_indices_2.append(idx)

    train_set_2 = Subset(dataset_2, train_indices_2)
    val_set_2 = Subset(dataset_2, val_indices_2)
    test_set_2 = Subset(dataset_2, test_indices_2)

    # CholecSeg8k Dataset

    dataset_3 = CholecSeg8kDataset(
        data_dir=r"D:\CholecSeg8k\cholecseg8k_tensors",
        history_length=4,
        augment=False
    )

    train_vids_3 = {}
    val_vids_3 = {}
    test_vids_3 = {1, 9, 12, 17, 18, 20, 24, 25, 26, 27, 28, 35, 37, 43, 48, 52, 55}

    train_indices_3 = []
    val_indices_3 = []
    test_indices_3 = []

    for idx, seq in enumerate(dataset_3.sequences):
        if isinstance(seq, tuple):
            files = seq[0]
        else:
            files = seq
        video_id = int(os.path.basename(files[0]).split('_')[0])
        if video_id in train_vids_3:
            train_indices_3.append(idx)
        elif video_id in val_vids_3:
            val_indices_3.append(idx)
        elif video_id in test_vids_3:
            test_indices_3.append(idx)

    train_set_3 = Subset(dataset_3, train_indices_3)
    val_set_3 = Subset(dataset_3, val_indices_3)
    test_set_3 = Subset(dataset_3, test_indices_3)

    # Concaténation des datasets de test
    test_set = ConcatDataset([test_set_1])
    test_loader = DataLoader(test_set, batch_size=2, shuffle=True, num_workers=4)

    model = DMBMSNet(
        input_dim=3,
        num_classes=8,
        num_encoder_blocks=5,
        base_dim=64,
        num_heads=8,
        k=5,
        memory_frames=4,
        num_memory_blocks=4,
        deformable=True,
        use_crf=True
    )

    model_dir = r"C:\Users\ewenr\Desktop\03 - SLU\DMBMS-Net\dresden_crf.pth"
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_model(
        model=model,
        test_loader=test_loader,
        num_classes=8,
        device="cuda",
        save_results=True,
        results_dir="results"
    )