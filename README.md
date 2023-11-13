![](assets/logo.png)

# FastSAM Segment Anything

[[`📕Paper`](https://arxiv.org/pdf/2306.12156.pdf)] [[`🤗HuggingFace Demo`](https://huggingface.co/spaces/An-619/FastSAM)] [[`Colab demo`](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9?usp=sharing)] [[`Replicate demo & API`](https://replicate.com/casia-iva-lab/fastsam)] [[`Model Zoo`](#model-checkpoints)] [[`BibTeX`](#citing-fastsam)]

![FastSAM Speed](assets/head_fig.png)

The **Fast Segment Anything Model(FastSAM)** is a CNN Segment Anything Model trained by only 2% of the SA-1B dataset published by SAM authors. The FastSAM achieve a comparable performance with
the SAM method at **50× higher run-time speed**.

![FastSAM design](assets/Overview.png)

**🍇 Refer from**
https://github.com/CASIA-IVA-Lab/FastSAM
[[`Original`]((https://github.com/CASIA-IVA-Lab/FastSAM)]


## Export ONNX to IR
```
    mo --input_model FastSAM-s.onnx --framework onnx
```

## Inference with Python

1. ```"cd FASTSAM_AWSOME_OPENVINO/src/Python"``` # change to python dir
2. ```"pip install -r requirements.txt"``` # install the requirements
3. ```"python FastSAM.py --model_path <model_path> --img_path <img_path>"``` # Inference


## Inference with cpp
Note: 
1. ```"cd FASTSAM_AWSOME_OPENVINO/src/CPlusPlus"```
2. Set OpenVINO_DIR in this CMakeLists.txt to your own openvino installation directory
3. ``` "mkdir build && cd build"```
4. ```"cmake .. && make -j4"```
 
![cat](outputs/cat.jpg)
![coco](outputs/coco.jpg)

## Reference
https://github.com/ChuRuaNh0/FastSam_Awsome_TensorRT
https://docs.openvino.ai/2023.1/home.html