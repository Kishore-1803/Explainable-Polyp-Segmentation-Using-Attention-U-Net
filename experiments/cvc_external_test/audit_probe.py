import torch, sys, numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path('src')))
from models.attention_unet import AttentionUNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]

state = torch.load('checkpoints/review_1/review_1_best.pth', map_location=DEVICE)
model = AttentionUNet(in_channels=3, out_channels=1, dropout_rate=0.0)
model.load_state_dict(state['model_state_dict'])
model.to(DEVICE).eval()

# Verify: what does checkpoint metadata say?
print("=== Checkpoint Metadata ===")
for k in state:
    if k not in ('model_state_dict', 'optimizer_state_dict'):
        print(f"  {k}: {state[k]}")

img_dir = Path('cvc_data/PNG/Original')
msk_dir = Path('cvc_data/PNG/Ground Truth')
imgs = sorted(img_dir.glob('*.png'), key=lambda x: int(x.stem))
msks = sorted(msk_dir.glob('*.png'), key=lambda x: int(x.stem))

print('\n=== Per-Image Inference Probe (10 samples) ===')
print(f'{"Img":<10} {"GT_fg%":<8} {"P_min":<8} {"P_max":<8} {"P_mean":<8} {"P_fg%":<8} {"Dice":<8}')

sample_indices = [0, 50, 100, 150, 200, 250, 300, 350, 400, 500]
for idx in sample_indices:
    img = Image.open(imgs[idx]).convert('RGB')
    img_t = TF.normalize(
        TF.to_tensor(TF.resize(img, (256,256), TF.InterpolationMode.BILINEAR)),
        MEAN, STD
    )
    msk = Image.open(msks[idx]).convert('L')
    msk_np = np.array(TF.resize(msk,(256,256),TF.InterpolationMode.NEAREST), dtype=np.float32)/255.0
    gt = (msk_np > 0.5).astype(np.float32)

    with torch.no_grad():
        logit = model(img_t.unsqueeze(0).to(DEVICE))
        prob = torch.sigmoid(logit)[0,0].cpu().numpy()

    pred_bin = (prob > 0.5)
    eps = 1e-7
    tp = np.sum(pred_bin & (gt > 0.5))
    fp = np.sum(pred_bin & (gt < 0.5))
    fn = np.sum(~pred_bin & (gt > 0.5))
    dice = (2*tp+eps)/(2*tp+fp+fn+eps)

    print(f'{imgs[idx].name:<10} {100*(gt>0.5).mean():<8.1f} {prob.min():<8.4f} {prob.max():<8.4f} {prob.mean():<8.4f} {100*pred_bin.mean():<8.1f} {dice:<8.4f}')

print('\n=== Mask Binarization Comparison for 5 noisy masks ===')
noisy_indices = [0, 1, 2, 3, 4]
for idx in noisy_indices:
    raw = np.array(Image.open(msks[idx]))[:,:,0]
    our   = (raw > 127).astype(np.uint8)
    nonzero = (raw > 0).astype(np.uint8)
    diff_pixels = abs(int(our.sum()) - int(nonzero.sum()))
    print(f'{msks[idx].name}: our_fg={our.sum()}, nonzero_fg={nonzero.sum()}, diff={diff_pixels} px ({100*diff_pixels/raw.size:.2f}%)')
