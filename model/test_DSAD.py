import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from DMBMSNet import DMBMSNet

def load_model(checkpoint_path, device="cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu")):
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
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def preprocess_image(image_path, img_size=(80, 80)):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img)

def load_ground_truth(image_path, img_size=(80, 80)):
    base_dir = os.path.dirname(image_path)
    frame_num = os.path.basename(image_path).replace("image", "").replace(".png", "")
    
    organs = [
        "abdominal_wall", "colon", "liver", "pancreas",
        "small_intestine", "spleen", "stomach"
    ]
    
    full_mask = np.zeros((8, *img_size), dtype=np.float32)
    
    for i, organ in enumerate(organs):
        mask_path = os.path.join(base_dir, f"mask{frame_num}_{organ}.png")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize(img_size, Image.NEAREST)
            full_mask[i] = np.array(mask) / 255.0
    
    full_mask[7] = (np.sum(full_mask[:7], axis=0) == 0).astype(np.float32)
    
    return torch.from_numpy(full_mask)

def visualize_comparison(image_tensor, pre_crf_mask, post_crf_mask, gt_mask):
    plt.figure(figsize=(24, 6))
    
    # Input Image
    plt.subplot(1, 4, 1)
    plt.imshow(image_tensor.permute(1, 2, 0).cpu())
    plt.title("Input Image")
    
    # Pre-CRF Prediction
    plt.subplot(1, 4, 2)
    pre_crf_class = pre_crf_mask.argmax(dim=0).cpu()
    plt.imshow(pre_crf_class, cmap="jet", vmin=0, vmax=7)
    plt.title("Pre-CRF Prediction")
    
    # Post-CRF Prediction
    plt.subplot(1, 4, 3)
    post_crf_class = post_crf_mask.argmax(dim=0).cpu()
    plt.imshow(post_crf_class, cmap="jet", vmin=0, vmax=7)
    plt.title("Post-CRF Prediction")
    
    # Ground Truth Overlay
    plt.subplot(1, 4, 4)
    plt.imshow(image_tensor.permute(1, 2, 0).cpu(), alpha=0.7)
    
    colors = plt.cm.get_cmap('tab10', 7)
    for i in range(7):
        mask = gt_mask[i].cpu().numpy()
        if np.any(mask > 0):
            plt.imshow(np.ma.masked_where(mask == 0, mask),
                      cmap=colors, alpha=0.5, vmin=0, vmax=1)
    
    plt.title("Ground Truth Overlay")
    plt.tight_layout()
    plt.show()

def main():
    device = "cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu")
    
    model = load_model(
        "D:/College/Research/SLU/laparoscopic surgery/LaparoSeg/DSAD_CRF_ON_best_model_checkpoint.pth",
        device
    )
    
    image_paths = [
        "D:/College/Research/SLU/laparoscopic surgery/Dataset/DSAD/multilabel/18/image10.png",
        "D:/College/Research/SLU/laparoscopic surgery/Dataset/DSAD/multilabel/30/image24.png",
        "D:/College/Research/SLU/laparoscopic surgery/Dataset/DSAD/multilabel/31/image61.png"
    ]
    
    for img_path in image_paths:
        print(f"\nProcessing: {img_path}")
        
        image_tensor = preprocess_image(img_path).to(device)
        gt_mask = load_ground_truth(img_path)
        
        memory_frames = [image_tensor] * 4
        
        with torch.no_grad():
            current_frame = image_tensor.unsqueeze(0)
            memory = torch.stack(memory_frames).unsqueeze(0)
            pre_crf, post_crf = model(current_frame, training=False, memory_frames=memory)
            pre_crf_mask = F.softmax(pre_crf[0], dim=0)
            post_crf_mask = F.softmax(post_crf[0], dim=0)
        
        visualize_comparison(image_tensor, pre_crf_mask, post_crf_mask, gt_mask)

if __name__ == "__main__":
    main()





























# import torch
# import matplotlib.pyplot as plt

# checkpoint_path = r"D:\College\Research\SLU\laparoscopic surgery\LaparoSeg\latest_checkpoint.pth"
# checkpoint = torch.load(checkpoint_path, map_location='xpu')

# train_loss_history = checkpoint["train_loss_history"]
# val_loss_history = checkpoint["val_loss_history"]

# plt.figure()
# plt.plot(range(1, len(train_loss_history)+1), train_loss_history, label="Training Loss")
# plt.plot(range(1, len(val_loss_history)+1), val_loss_history, label="Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training and Validation Loss Evolution")
# plt.legend()
# plt.show()