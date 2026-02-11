
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


def preprocessing(image1_file, image2_file, is_test=True, size=(448,448)):
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



import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', '-r', help='experiment configure file name', default=512, type=int)
    parser.add_argument('--ckpt', help='checkpoint path', type=str)
    parser.add_argument('--model', '-m', help='checkpoint path', default='S', type=str)
    args = parser.parse_args()

    resolution = args.resolution


    image1_file = 'assets/frame_0016.png'
    image2_file = 'assets/frame_0018.png'
    image1, image2 = preprocessing(image1_file, image2_file, size=(resolution, resolution))

    from model.raft.raft import RAFT

    from types import SimpleNamespace


    conf = {
        "cfg": 'config/raft/eval/sintel-S.json',
        "url": None,
        'path': 'weights/raft/Tartan-C-T-TSKH432x960-S.pth',
        "device": 'cpu',
        }
    # conf = {}
    # args = SimpleNamespace(**{**default_conf, **conf})

    from config.parser import json_to_args
    from utils.utils import load_ckpt, coords_grid, bilinear_sampler

    opts =json_to_args(f'config/raft/eval/sintel-{args.model}.json')
    opts.path = 'weights/raft/Tartan-C-T-TSKH432x960-S.pth'
    model = RAFT(opts)
    load_ckpt(model, opts.path)

    model = model.eval().cuda()

    out = model.export(image1.cuda(), image2.cuda())




    # data = {"image1": image1.cpu(), "image2": image2.cpu()}
    # model = model.eval().cpu()
    # model.forward = model.export
    # export2onnx(model, data, f'sea_raft_{args.model}_{resolution}.onnx')


if __name__ == '__main__':
    main()
