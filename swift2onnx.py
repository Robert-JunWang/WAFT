
import torch
import onnx
def export2onnx(model, dummy_input, out_file):
    with torch.no_grad():
        torch.onnx.export(model, dummy_input, 
                        out_file, 
                        verbose=True, opset_version=17)

    onnx_model = onnx.load(out_file)
    onnx.checker.check_model(onnx_model)

    try:
        import onnxsim
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, 'assert check failed'
    except Exception as e:
        print(f'Simplify failure: {e}')
    onnx.save(onnx_model, out_file)
    print(f'ONNX export success, save into {out_file}')


from utils import frame_utils
import numpy as np
import torch, cv2

def preprocessing(image1_file, image2_file, is_test=True, size=(512,512)):
        if is_test:
            img1 = frame_utils.read_gen(image1_file)
            img2 = frame_utils.read_gen(image2_file)
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = cv2.resize(img1, size) 
            img2 = cv2.resize(img2, size) 
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0)
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0)
            return img1, img2



##################################################
    
import argparse, json


from model import fetch_model
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt, coords_grid, bilinear_sampler


cfg = 'config/swift/dinov3/chairs.json'
# ckpt = 'weights/a2/waftv2-ckpts/dinov3/sintel.pth'

args = argparse.Namespace()
args_dict = args.__dict__

# args_dict['ckpt'] = ckpt
args_dict['scale'] = 0.0

with open(cfg, 'r') as f:
    data = json.load(f)

    for key, value in data.items():
        args_dict[key] = value


model = fetch_model(args)
# load_ckpt(model, args.ckpt)
model = model.cuda()
model.eval()
# wrapped_model = InferenceWrapper(model, scale=args.scale, train_size=args.image_size, pad_to_train_size=False, tiling=False)

image1_file = 'assets/frame_0016.png'
image2_file = 'assets/frame_0018.png'
size = 512
image1, image2 = preprocessing(image1_file, image2_file, size=(size, size))


image1 = model.normalize_image(image1).cuda()
image2 = model.normalize_image(image2).cuda()

out = model.export(image1, image2)
data = {"image1": image1.cpu(), "image2": image2.cpu()}
model = model.eval().cpu()
model.forward = model.export
export2onnx(model, data, f'onnx/swift_512_coarse_256.onnx')