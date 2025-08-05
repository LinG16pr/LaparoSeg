from scipy.ndimage import distance_transform_edt, binary_erosion
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import os
import seaborn as sns

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

def test_model(model, test_loader, num_classes=9, device="cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu"), save_results=False, results_dir="results"):
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
    results_per_image = []    # liste de dictionnaires, chacun contenant les métriques pour une image
    # Pour l'agrégation par classe, on utilisera des dictionnaires listant les valeurs uniquement pour les images qui possèdent la classe
    agg_metrics_pre = {
        "accuracy": {c: [] for c in range(num_classes)},
        "dice": {c: [] for c in range(num_classes)},
        "jaccard": {c: [] for c in range(num_classes)},
        "hd95": {c: [] for c in range(num_classes)},
    }
    agg_metrics_post = {
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

            outputs = model(current_frame, training=False, memory_frames=memory_frames, memory_masks=memory_masks)
            
            if model.use_crf:
                pre_crf_pred, post_crf_pred = outputs
                loss = criterion(post_crf_pred, current_mask.argmax(dim=1).long())
            else:
                post_crf_pred = outputs
                loss = criterion(post_crf_pred, current_mask.argmax(dim=1).long())

            print(f"[{image_counter+1}/{len(test_loader)*len(post_crf_pred)}] Testing loss: {loss.item():.4f}")
            test_loss += loss.item()

            # Conversion des masques en arrays numpy
            target_masks = current_mask.argmax(dim=1).detach().cpu().numpy()
            
            if model.use_crf:
                # Process pre-CRF results
                pre_crf_masks = pre_crf_pred.argmax(dim=1).detach().cpu().numpy()
            
            # Process post-CRF results
            post_crf_masks = post_crf_pred.argmax(dim=1).detach().cpu().numpy()

            # Calculate for each image in batch
            for i in range(post_crf_masks.shape[0]):
                image_counter += 1
                image_results = {"Image_Index": image_counter}

                if model.use_crf:
                    # Pre-CRF metrics
                    pre_acc = accuracy_score(pre_crf_masks[i], target_masks[i])
                    pre_dice = dice_score_per_sample(pre_crf_masks[i], target_masks[i], num_classes)
                    pre_jaccard = jaccard_score_per_sample(pre_crf_masks[i], target_masks[i], num_classes)
                    pre_hd95 = hausdorff95_score_per_sample(pre_crf_masks[i], target_masks[i], num_classes)
                
                    image_results.update({
                        "Pre_CRF_Accuracy": pre_acc,
                        "Pre_CRF_Dice": pre_dice,
                        "Pre_CRF_Jaccard": pre_jaccard,
                        "Pre_CRF_Hd95": pre_hd95
                    })

                    for c in range(num_classes):
                        if np.sum(target_masks[i] == c) > 0:
                            acc_pc = accuracy_score_per_class(pre_crf_masks[i], target_masks[i], num_classes)[c]
                            dice_pc = dice_score_per_class(pre_crf_masks[i], target_masks[i], num_classes)[c]
                            jaccard_pc = jaccard_score_per_class(pre_crf_masks[i], target_masks[i], num_classes)[c]
                            hd95_pc = hausdorff95_score_per_class(pre_crf_masks[i], target_masks[i], num_classes)[c]

                            image_results.update({
                                f"Pre_CRF_Accuracy_{class_names[c]}": acc_pc,
                                f"Pre_CRF_Dice_{class_names[c]}": dice_pc,
                                f"Pre_CRF_Jaccard_{class_names[c]}": jaccard_pc,
                                f"Pre_CRF_Hd95_{class_names[c]}": hd95_pc
                            })

                            agg_metrics_pre["accuracy"][c].append(acc_pc)
                            agg_metrics_pre["dice"][c].append(dice_pc)
                            agg_metrics_pre["jaccard"][c].append(jaccard_pc)
                            agg_metrics_pre["hd95"][c].append(hd95_pc)
            
                post_acc = accuracy_score(post_crf_masks[i], target_masks[i])
                post_dice = dice_score_per_sample(post_crf_masks[i], target_masks[i], num_classes)
                post_jaccard = jaccard_score_per_sample(post_crf_masks[i], target_masks[i], num_classes)
                post_hd95 = hausdorff95_score_per_sample(post_crf_masks[i], target_masks[i], num_classes)
                
                image_results.update({
                    "Post_CRF_Accuracy": post_acc,
                    "Post_CRF_Dice": post_dice,
                    "Post_CRF_Jaccard": post_jaccard,
                    "Post_CRF_Hd95": post_hd95
                })

                for c in range(num_classes):
                    if np.sum(target_masks[i] == c) > 0:
                        acc_pc = accuracy_score_per_class(post_crf_masks[i], target_masks[i], num_classes)[c]
                        dice_pc = dice_score_per_class(post_crf_masks[i], target_masks[i], num_classes)[c]
                        jaccard_pc = jaccard_score_per_class(post_crf_masks[i], target_masks[i], num_classes)[c]
                        hd95_pc = hausdorff95_score_per_class(post_crf_masks[i], target_masks[i], num_classes)[c]
                        
                        image_results.update({
                            f"Post_CRF_Accuracy_{class_names[c]}": acc_pc,
                            f"Post_CRF_Dice_{class_names[c]}": dice_pc,
                            f"Post_CRF_Jaccard_{class_names[c]}": jaccard_pc,
                            f"Post_CRF_Hd95_{class_names[c]}": hd95_pc
                        })
                        
                        agg_metrics_post["accuracy"][c].append(acc_pc)
                        agg_metrics_post["dice"][c].append(dice_pc)
                        agg_metrics_post["jaccard"][c].append(jaccard_pc)
                        agg_metrics_post["hd95"][c].append(hd95_pc)
                
                results_per_image.append(image_results)

            del current_frame, current_mask, memory_frames, memory_masks, outputs, loss
            if model.use_crf:
                del pre_crf_pred, post_crf_pred
            else:
                del post_crf_pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.xpu.is_available():
                torch.xpu.empty_cache()

    avg_test_loss = test_loss / len(test_loader)
    print("------------------------------")
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Calculate global metrics
    global_metrics = {}
    if model.use_crf:
        all_pre_acc = [d["Pre_CRF_Accuracy"] for d in results_per_image]
        all_pre_dice = [d["Pre_CRF_Dice"] for d in results_per_image]
        all_pre_jaccard = [d["Pre_CRF_Jaccard"] for d in results_per_image]
        all_pre_hd95 = [d["Pre_CRF_Hd95"] for d in results_per_image if np.isfinite(d["Pre_CRF_Hd95"])]

        global_metrics["pre_crf"] = {
            "accuracy": np.mean(all_pre_acc),
            "dice": np.mean(all_pre_dice),
            "jaccard": np.mean(all_pre_jaccard),
            "hd95": np.mean(all_pre_hd95) if len(all_pre_hd95) > 0 else np.inf
        }

        print("Pre-CRF Metrics:")
        print(f"Global Accuracy: {global_metrics['pre_crf']['accuracy']:.4f}")
        print(f"Global Dice Coefficient: {global_metrics['pre_crf']['dice']:.4f}")
        print(f"Global Jaccard Index: {global_metrics['pre_crf']['jaccard']:.4f}")
        print(f"Global Hausdorff Distance 95: {global_metrics['pre_crf']['hd95']:.4f}")
        print("------------------------------")

    all_post_acc = [d["Post_CRF_Accuracy"] for d in results_per_image]
    all_post_dice = [d["Post_CRF_Dice"] for d in results_per_image]
    all_post_jaccard = [d["Post_CRF_Jaccard"] for d in results_per_image]
    all_post_hd95 = [d["Post_CRF_Hd95"] for d in results_per_image if np.isfinite(d["Post_CRF_Hd95"])]

    global_metrics["post_crf"] = {
        "accuracy": np.mean(all_post_acc),
        "dice": np.mean(all_post_dice),
        "jaccard": np.mean(all_post_jaccard),
        "hd95": np.mean(all_post_hd95) if len(all_post_hd95) > 0 else np.inf
    }
    print("Post-CRF Metrics:")
    print(f"Global Accuracy: {global_metrics['post_crf']['accuracy']:.4f}")
    print(f"Global Dice Coefficient: {global_metrics['post_crf']['dice']:.4f}")
    print(f"Global Jaccard Index: {global_metrics['post_crf']['jaccard']:.4f}")
    print(f"Global Hausdorff Distance 95: {global_metrics['post_crf']['hd95']:.4f}")
    print("------------------------------")


    # Calcul des moyennes par classe pour les images qui contiennent chacune la classe
    avg_per_class_pre = {}
    avg_per_class_post = {}
    for c in range(num_classes):
        if model.use_crf and agg_metrics_pre["accuracy"][c]:
            avg_per_class_pre[class_names[c]] = {
                "Accuracy": np.mean(agg_metrics_pre["accuracy"][c]),
                "Dice": np.mean(agg_metrics_pre["dice"][c]),
                "Jaccard": np.mean(agg_metrics_pre["jaccard"][c]),
                "Hd95": np.mean([v for v in agg_metrics_pre["hd95"][c] if np.isfinite(v)]) if any(np.isfinite(agg_metrics_pre["hd95"][c])) else np.inf
            }
        
        if agg_metrics_post["accuracy"][c]:
            avg_per_class_post[class_names[c]] = {
                "Accuracy": np.mean(agg_metrics_post["accuracy"][c]),
                "Dice": np.mean(agg_metrics_post["dice"][c]),
                "Jaccard": np.mean(agg_metrics_post["jaccard"][c]),
                "Hd95": np.mean([v for v in agg_metrics_post["hd95"][c] if np.isfinite(v)]) if any(np.isfinite(agg_metrics_post["hd95"][c])) else np.inf
            }

    if model.use_crf:
        print("Pre-CRF Metrics per class:")
        for cl, metrics in avg_per_class_pre.items():
            print(f"{cl}:")
            print(f"   Accuracy: {metrics['Accuracy']:.4f}")
            print(f"   Dice Coefficient: {metrics['Dice']:.4f}")
            print(f"   Jaccard Index: {metrics['Jaccard']:.4f}")
            print(f"   Hausdorff Distance 95: {metrics['Hd95']:.4f}")
        print("------------------------------")

    print("Post-CRF Metrics per class:")
    for cl, metrics in avg_per_class_post.items():
        print(f"{cl}:")
        print(f"   Accuracy: {metrics['Accuracy']:.4f}")
        print(f"   Dice Coefficient: {metrics['Dice']:.4f}")
        print(f"   Jaccard Index: {metrics['Jaccard']:.4f}")
        print(f"   Hausdorff Distance 95: {metrics['Hd95']:.4f}")
    print("------------------------------")

    # Sauvegarde des résultats dans des fichiers CSV si demandé
    if save_results:
        viz_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        df_images = pd.DataFrame(results_per_image)
        df_images.to_csv(os.path.join(results_dir, "per_image_metrics.csv"), index=False)
        print(f"Résultats par image sauvegardés dans {os.path.join(results_dir, 'per_image_metrics.csv')}")
        
        global_data = {
            "test_loss": [avg_test_loss]
        }
        
        if model.use_crf:
            global_data.update({
                "pre_crf_accuracy": [global_metrics['pre_crf']['accuracy']],
                "pre_crf_dice": [global_metrics['pre_crf']['dice']],
                "pre_crf_jaccard": [global_metrics['pre_crf']['jaccard']],
                "pre_crf_hd95": [global_metrics['pre_crf']['hd95']]
            })
        
        global_data.update({
            "post_crf_accuracy": [global_metrics['post_crf']['accuracy']],
            "post_crf_dice": [global_metrics['post_crf']['dice']],
            "post_crf_jaccard": [global_metrics['post_crf']['jaccard']],
            "post_crf_hd95": [global_metrics['post_crf']['hd95']]
        })
        
        df_global = pd.DataFrame(global_data)
        df_global.to_csv(os.path.join(results_dir, "global_metrics.csv"), index=False)
        
        class_data = []
        for class_name in class_names:
            if model.use_crf and class_name in avg_per_class_pre:
                class_data.append({
                    "Class": class_name,
                    "Type": "Pre-CRF",
                    "Accuracy": avg_per_class_pre[class_name]["Accuracy"],
                    "Dice": avg_per_class_pre[class_name]["Dice"],
                    "Jaccard": avg_per_class_pre[class_name]["Jaccard"],
                    "Hd95": avg_per_class_pre[class_name]["Hd95"]
                })
            
            if class_name in avg_per_class_post:
                class_data.append({
                    "Class": class_name,
                    "Type": "Post-CRF",
                    "Accuracy": avg_per_class_post[class_name]["Accuracy"],
                    "Dice": avg_per_class_post[class_name]["Dice"],
                    "Jaccard": avg_per_class_post[class_name]["Jaccard"],
                    "Hd95": avg_per_class_post[class_name]["Hd95"]
                })
        
        df_class = pd.DataFrame(class_data)
        df_class.to_csv(os.path.join(results_dir, "per_class_metrics.csv"), index=False)
        
        print(f"Résultats globaux sauvegardés dans {results_dir}")
        
        import matplotlib.pyplot as plt
        
        metrics_to_plot = ['Accuracy', 'Dice', 'Jaccard', 'Hd95']
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            
            if model.use_crf:
                pre_values = [d[f"Pre_CRF_{metric}"] for d in results_per_image]
                plt.hist(pre_values, bins=20, alpha=0.5, label='Pre-CRF')
            
            post_values = [d[f"Post_CRF_{metric}"] for d in results_per_image]
            plt.hist(post_values, bins=20, alpha=0.5, label='Post-CRF')
            
            plt.title(f'Distribution of {metric} Scores')
            plt.xlabel(metric)
            plt.ylabel('Number of Images')
            plt.legend()
            plt.grid(True)
            
            if metric == 'Hd95':
                plt.xlim(0, np.percentile([v for v in post_values if np.isfinite(v)], 95))
            
            plt.savefig(os.path.join(viz_dir, f'{metric.lower()}_distribution.png'))
            plt.close()
        
        for class_name in class_names:
            for metric in metrics_to_plot:
                plt.figure(figsize=(10, 6))
                
                if model.use_crf:
                    pre_values = [d[f"Pre_CRF_{metric}"] for d in results_per_image]
                    pre_values = [v for v in pre_values if np.isfinite(v) and not np.isnan(v)]
                    if pre_values:
                        plt.hist(pre_values, bins=20, alpha=0.5, label='Pre-CRF')
                
                post_values = [d.get(f"Post_CRF_{metric}_{class_name}", np.nan) for d in results_per_image]
                post_values = [v for v in post_values if np.isfinite(v) and not np.isnan(v)]
                if post_values:
                    plt.hist(post_values, bins=20, alpha=0.5, label='Post-CRF')
                
                if (model.use_crf and pre_values) or post_values:
                    plt.title(f'Distribution of {metric} Scores for {class_name}')
                    plt.xlabel(metric)
                    plt.ylabel('Number of Images')
                    plt.legend()
                    plt.grid(True)
                    
                    if metric == 'Hd95':
                        finite_values = [v for v in post_values if np.isfinite(v)]
                        if finite_values:
                            plt.xlim(0, np.percentile(finite_values, 95))
                    
                    plt.savefig(os.path.join(viz_dir, f'{class_name.lower().replace(" ", "_")}_{metric.lower()}_distribution.png'))
                    plt.close()

    return {
        "test_loss": avg_test_loss,
        "global_metrics": global_metrics,
        "per_class_pre_crf": avg_per_class_pre if model.use_crf else None,
        "per_class_post_crf": avg_per_class_post,
        "per_image_metrics": results_per_image
    }


if __name__ == '__main__':

    # Dresden Dataset

    # dataset_1 = DresdenDataset(
    #     data_dir=r"D:\College\Research\SLU\laparoscopic surgery\Dataset\DSAD\multilabel",
    #     history_length=4,
    #     augment=False
    # )

    # train_vids_1 = {4, 5, 8, 10, 12, 16, 17, 22, 23, 24, 25, 27, 29, 30, 31}
    # val_vids_1   = {2, 7, 11, 18, 20}
    # test_vids_1  = {3, 21, 26}

    # train_indices_1 = []
    # val_indices_1 = []
    # test_indices_1 = []

    # for idx, seq in enumerate(dataset_1.sequences):
    #     if isinstance(seq, tuple):
    #         files = seq[0]
    #     else:
    #         files = seq
    #     video_id = files[0]["surgery_id"]
    #     if video_id in train_vids_1:
    #         train_indices_1.append(idx)
    #     elif video_id in val_vids_1:
    #         val_indices_1.append(idx)
    #     elif video_id in test_vids_1:
    #         test_indices_1.append(idx)

    # train_set_1 = Subset(dataset_1, train_indices_1)
    # val_set_1 = Subset(dataset_1, val_indices_1)
    # test_set_1 = Subset(dataset_1, test_indices_1)

    # HemoSet Dataset

    dataset_2 = HemoSetDataset(
        data_dir=r"D:\College\Research\SLU\laparoscopic surgery\Dataset\HemoSet",
        history_length=3,
        augment=False,
        skip_rate=5
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
        video_id = files[0]["pig_id"]
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

    # dataset_3 = CholecSeg8kDataset(
    #     data_dir=r"D:\College\Research\SLU\laparoscopic surgery\Dataset\CholecSeg8k",
    #     history_length=4,
    #     augment=False
    # )

    # train_vids_3 = {1, 9, 12, 17, 18, 20, 24, 25, 26, 27, 28}
    # val_vids_3 = {35, 37, 43}
    # test_vids_3 = {48, 52, 55}

    # train_indices_3 = []
    # val_indices_3 = []
    # test_indices_3 = []

    # for idx, seq in enumerate(dataset_3.sequences):
    #     if isinstance(seq, tuple):
    #         files = seq[0]
    #     else:
    #         files = seq
    #     video_id = files[0]["video_id"]
    #     if video_id in train_vids_3:
    #         train_indices_3.append(idx)
    #     elif video_id in val_vids_3:
    #         val_indices_3.append(idx)
    #     elif video_id in test_vids_3:
    #         test_indices_3.append(idx)

    # train_set_3 = Subset(dataset_3, train_indices_3)
    # val_set_3 = Subset(dataset_3, val_indices_3)
    # test_set_3 = Subset(dataset_3, test_indices_3)

    # Concaténation des datasets de test
    test_set = ConcatDataset([test_set_2])
    test_loader = DataLoader(test_set, batch_size=2, shuffle=True, num_workers=4)

    model = DMBMSNet(
        input_dim=3,
        num_classes=2,
        num_encoder_blocks=5,
        base_dim=64,
        num_heads=8,
        k=5,
        memory_frames=4,
        num_memory_blocks=4,
        deformable=True,
        use_crf=True
    )

    model_dir = r"D:\College\Research\SLU\laparoscopic surgery\LaparoSeg\Hemo_Memory_3_Skip_Rate_0_CRF_best_model_checkpoint.pth"
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_model(
        model=model,
        test_loader=test_loader,
        num_classes=2,
        device="cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu"),
        save_results=True,
        results_dir="results"
    )