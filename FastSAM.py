import time
from typing import Any
import cv2
import numpy as np
import openvino as ov
import os
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops
import torch
from random import randint, choice
import string
import argparse

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(choice(characters) for _ in range(length))
    return random_string

class FastSAM:
    
    # 
    def __init__(self, model_path: str, conf_thres:float = 0.4, iou_thres:float = 0.9, device: str = "AUTO", outputPath: str = "./outputs") -> None:
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.device = device
        self.outputPath = outputPath
        self.InitializeModel(model_path=model_path)


    def InitializeModel(self, model_path: str):
        core = ov.Core()

        model = core.read_model(model=model_path)
        self.compiled_model = core.compile_model(model = model, device_name=self.device)
        
        self.GetInputDetail()
        self.GetOutputDetail()
    
    def GetInputDetail(self):
        # 输入一般只有一个维度，即这个inputs 长度为1
        self.inputs = self.compiled_model.inputs
        

    def GetOutputDetail(self):
        self.outputs = self.compiled_model.outputs

    def __call__(self, image_path: str) -> Any:
        if not os.path.exists(image_path):
            print("The image path is not exist! ", image_path)
            return None
        
        return self.__call__(cv2.imread(image_path))

    def __call__(self, image: cv2.Mat) -> Any:
        return self.SegmentObjects(OriginalImage=image)

   
    def overlay(self, image: np.ndarray, mask: np.ndarray, color: tuple, alpha: float, resize=None):
        """Combines image and its segmentation mask into a single image.
        https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

        Params:
            image: Training image. np.ndarray,
            mask: Segmentation mask. np.ndarray,
            color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
            alpha: Segmentation mask's transparency. float = 0.5,
            resize: If provided, both image and its mask are resized before blending them together.
            tuple[int, int] = (1024, 1024))

        Returns:
            image_combined: The combined image. np.ndarray

        """
        color = color[::-1]
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        if resize is not None:
            image = cv2.resize(image.transpose(1, 2, 0), resize)
            image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

        return cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    """
        targetShape: 需要变成的图像宽高
    """
    def Preprocess(self, image: cv2.Mat, targetShape: list):
        th, tw = targetShape
        h, w = image.shape[:2]
        if h>w:
            scale   = min(th / h, tw / w)
            inp     = np.zeros((th, tw, 3), dtype = np.uint8)
            nw      = int(w * scale)
            nh      = int(h * scale)
            a = int((nh-nw)/2) 
            inp[: nh, a:a+nw, :] = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (nw, nh))
        else:
            scale   = min(th / h, tw / w)
            inp     = np.zeros((th, tw, 3), dtype = np.uint8)
            nw      = int(w * scale)
            nh      = int(h * scale)
            a = int((nw-nh)/2) 

            inp[a: a+nh, :nw, :] = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (nw, nh))
        rgb = np.array([inp], dtype = np.float32) / 255.0
        return np.transpose(rgb, (0, 3, 1, 2)) # 重新排列为batch_size, channels, height, width

    
    def Postprocess(self, preds, img, orig_imgs, retina_masks, conf, iou, agnostic_nms=False):
        p = ops.non_max_suppression(preds[0],
                                conf,
                                iou,
                                agnostic_nms,
                                max_det=100,
                                nc=1)
        results = []
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            # path = self.batch[0]
            img_path = "ok"
            if not len(pred):  # save empty boxes
                results.append(Results(orig_img=orig_img, path=img_path, names="segment", boxes=pred[:, :6]))
                continue
            if retina_masks:
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(
                Results(orig_img=orig_img, path=img_path, names="1213", boxes=pred[:, :6], masks=masks))
        return results

    # 实际分割地方
    def SegmentObjects(self, OriginalImage: cv2.Mat):
        N, C, H, W = self.inputs[0].shape
        Size = [H,W]
        input = self.Preprocess(OriginalImage, Size)

        start = time.time()
        result = self.compiled_model([input])
        end = time.time()
        print("inference time:{} ms".format((end - start) * 1000) )

        output1 = torch.from_numpy(result[self.outputs[0]])
        output2 = torch.from_numpy(result[self.outputs[1]])
        pred = [output1, output2]

        ans = self.Postprocess(preds=pred, img=input, orig_imgs=OriginalImage, retina_masks=True, conf=self.conf_threshold, iou=self.iou_threshold)

        masks = ans[0].masks.data

        image_with_masks = np.copy(OriginalImage)
        for i, mask_i in enumerate(masks):
            r = randint(0, 255)
            g = randint(0, 255)
            b = randint(0, 255)
            rand_color = (r, g, b)
            image_with_masks = self.overlay(image_with_masks, mask_i, color=rand_color, alpha=1)

        outPath = self.outputPath
        if not os.path.exists(outPath):
            os.makedirs(outPath)

        outFilePath = outPath + generate_random_string(10) + ".jpg"
        print("out path:", outFilePath)

        cv2.imwrite(outFilePath, image_with_masks)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./models/FastSAM-s.xml", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="./images/coco.jpg", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./outputs/", help="image save path"
    )

    parser.add_argument(
        "--device", type=str, default="GPU", help="[GPU] or [CPU] or [AUTO]"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    infer = FastSAM(model_path=args.model_path, conf_thres=args.conf, iou_thres=args.iou, device=args.device, outputPath=args.output)

    image = cv2.imread(args.img_path)
    infer(image)

   