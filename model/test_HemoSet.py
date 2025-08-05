import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from DMBMSNet import DMBMSNet
import os
import glob

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

num_classes = 2
num_encoder_blocks = 5
num_heads = 8
k = 5
memory_length = 4
num_memory_blocks = 4
deformable = True
use_crf = True

model = DMBMSNet(
    input_dim=3,
    num_classes=num_classes,
    num_encoder_blocks=num_encoder_blocks,
    base_dim=64,
    num_heads=num_heads,
    k=k,
    memory_frames=memory_length,
    num_memory_blocks=num_memory_blocks,
    deformable=deformable,
    use_crf=use_crf
).to(device)

checkpoint = torch.load('D:\College\Research\SLU\laparoscopic surgery\LaparoSeg\Hemo_Use_CRF_Freeze_others_reset_and_retrain_CRF_best_model_checkpoint_LR_1e-3.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)

def load_mask(mask_path):
    mask = np.array(Image.open(mask_path).convert("RGB"))
    blood_mask = (mask[..., 0] > 110) & (mask[..., 1] == 0) & (mask[..., 2] == 0)
    full_mask = np.zeros((2, *mask.shape[:2]), dtype=np.float32)
    full_mask[0] = blood_mask.astype(np.float32)
    full_mask[1] = 1.0 - full_mask[0]
    return torch.from_numpy(full_mask)

def get_sequence_frames(base_path, frame_num, history_length=4):
    """Get previous frames in sequence for memory processing"""
    dir_path = os.path.dirname(base_path)
    base_name = os.path.basename(base_path).split('.')[0]
    
    all_frames = sorted(glob.glob(os.path.join(dir_path, "*.png")))
    
    frame_numbers = []
    for f in all_frames:
        try:
            num = int(os.path.basename(f).split('.')[0])
            frame_numbers.append(num)
        except:
            continue
    
    try:
        current_idx = frame_numbers.index(frame_num)
    except ValueError:
        print(f"Frame {frame_num} not found in sequence")
        return []
    
    start_idx = max(0, current_idx - history_length)
    
    return [all_frames[i] for i in range(start_idx, current_idx + 1)]

def visualize_results(img_path, pre_crf_pred, post_crf_pred, mask_path=None):
    plt.figure(figsize=(20, 5))
    
    # Input Image
    plt.subplot(1, 4, 1)
    plt.imshow(Image.open(img_path).resize((1024, 1024)))
    plt.title('Input Image')
    plt.axis('off')
    
    # Pre-CRF Prediction
    plt.subplot(1, 4, 2)
    plt.imshow(pre_crf_pred == 0, cmap='Reds')
    plt.title('Pre-CRF Prediction')
    plt.axis('off')
    
    # Post-CRF Prediction
    plt.subplot(1, 4, 3)
    plt.imshow(post_crf_pred == 0, cmap='Reds')
    plt.title('Post-CRF Prediction')
    plt.axis('off')
    
    # Ground Truth
    if mask_path:
        plt.subplot(1, 4, 4)
        plt.imshow(Image.open(mask_path).resize((1024, 1024)))
        plt.title('Ground Truth')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{os.path.basename(img_path).split('.')[0]}_result.png")
    plt.show()

def test_sequence(current_frame_path, mask_path=None):
    """Test with actual sequence of frames for memory processing"""
    frame_num = int(os.path.basename(current_frame_path).split('.')[0])
    
    frame_paths = get_sequence_frames(current_frame_path, frame_num, memory_length)
    
    if not frame_paths:
        print("Couldn't find sequence frames, falling back to single frame")
        test_single_frame(current_frame_path, mask_path)
        return
    
    frames = [preprocess_image(fp) for fp in frame_paths]
    current_frame = frames[-1]
    memory_frames = torch.cat(frames[:-1], dim=0)
    
    if mask_path:
        mask_dir = os.path.dirname(mask_path)
        mask_paths = [fp.replace("imgs", "labels").replace(".png", "_mask.png") for fp in frame_paths]
        masks = [load_mask(mp) for mp in mask_paths if os.path.exists(mp)]
        if masks:
            memory_masks = torch.stack(masks[:-1], dim=0).to(device)
        else:
            memory_masks = torch.zeros((len(frames)-1, num_classes, 80, 80)).to(device)
    else:
        memory_masks = torch.zeros((len(frames)-1, num_classes, 80, 80)).to(device)
    
    with torch.no_grad():
        pre_crf_output, post_crf_output = model(
            current_frame,
            training=False,
            memory_frames=memory_frames,
            memory_masks=memory_masks
        )
    
    pre_crf_pred = torch.argmax(pre_crf_output.squeeze(), dim=0).cpu().numpy()
    post_crf_pred = torch.argmax(post_crf_output.squeeze(), dim=0).cpu().numpy()
    
    visualize_results(current_frame_path, pre_crf_pred, post_crf_pred, mask_path)

def test_single_frame(img_path, mask_path=None):
    """Test with single frame (no memory processing)"""
    current_frame = preprocess_image(img_path)
    
    with torch.no_grad():
        pre_crf_output, post_crf_output = model(
            current_frame,
            training=False
        )
    
    pre_crf_pred = torch.argmax(pre_crf_output.squeeze(), dim=0).cpu().numpy()
    post_crf_pred = torch.argmax(post_crf_output.squeeze(), dim=0).cpu().numpy()
    
    visualize_results(img_path, pre_crf_pred, post_crf_pred, mask_path)

test_images = [
    ("D:/College/Research/SLU/laparoscopic surgery/Dataset/HemoSet/pig3/imgs/002520.png",
     "D:/College/Research/SLU/laparoscopic surgery/Dataset/HemoSet/pig3/labels/002520_mask.png"),
    ("D:/College/Research/SLU/laparoscopic surgery/Dataset/HemoSet/pig5/imgs/006840.png",
     "D:/College/Research/SLU/laparoscopic surgery/Dataset/HemoSet/pig5/labels/006840_mask.png"),
    ("D:/College/Research/SLU/laparoscopic surgery/Dataset/HemoSet/pig7/imgs/003180.png",
     "D:/College/Research/SLU/laparoscopic surgery/Dataset/HemoSet/pig7/labels/003180_mask.png"),
    ("D:/College/Research/SLU/laparoscopic surgery/Dataset/HemoSet/pig11/imgs/000960.png",
     "D:/College/Research/SLU/laparoscopic surgery/Dataset/HemoSet/pig11/labels/000960_mask.png"),
    ("D:/College/Research/SLU/laparoscopic surgery/Dataset/HemoSet/pig10/imgs/004560.png",
     "D:/College/Research/SLU/laparoscopic surgery/Dataset/HemoSet/pig10/labels/004560_mask.png")
]

for img_path, mask_path in test_images:
    print(f"\nTesting sequence starting with: {img_path}")
    if memory_length == 0:
        test_single_frame(img_path, mask_path)
    else:
        test_sequence(img_path, mask_path)