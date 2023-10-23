from ultralytics import YOLO
# from utils.tools import *
import argparse
# from models.experimental import attempt_load
import torch.nn as nn
import torch

# reference: https://github.com/ChuRuaNh0/FastSam_Awsome_TensorRT
class FastSamAddNMS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, input):
        """ 
            Split output [n_batch, 84, n_bboxes] to 3 output: bboxes, scores, classes
        """ 
        # x, y, w, h -> x1, y1, x2, y2
        output = self.model(input)
        print('Output: ', len(output))
        output = output[0]
        print(output.shape)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./models/FastSAM-s.pt', help='weights path')
    # parser.add_argument('--cfg', type=str, default='cfg/yolo_nas.cfg', help='config path')
    parser.add_argument('--output', type=str, default='./models/FastSAM-s.onnx', help='output ONNX model path')
    parser.add_argument('--max_size', type=int, default=416, help='max size of input image')
    opt = parser.parse_args()

    # model_cfg = opt.cfg
    model_weights = opt.weights
    output_model_path = opt.output
    max_size = opt.max_size
    device = torch.device("cuda")

    # load model 
    print("[Info] Load Model")
    # model = attempt_load(model_weights, device=device, inplace=True, fuse=True)
    model_ = YOLO(model_weights)
    
    model = model_.model
    

    img = torch.zeros(1, 3, max_size, max_size).to(device)

    
    print("[Info] Preprocess Model")
    # model = FastSamAddNMS(model)
    # exit(1)
    output_names = ['output0', 'output1'] #if isinstance(model, SegmentationModel) else ['output0']
    dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
    dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
    dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)

    model.eval().to(device)

    print('[INFO] Convert from Torch to ONNX')

    torch.onnx.export(model,               # model being run
                    img,                         # model input (or a tuple for multiple inputs)
                    output_model_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['images'],   # the model's input names
                    output_names = output_names, # the model's output names
                    dynamic_axes=dynamic)

    print('[INFO] Finished Convert!')