# YOLO_DeepSORT_TensorRT_Cpp


# nabYolov3DeepsortTensort 

```
├──nabTensorRTxCode (github)
│  └── yolov3 - nabYolov3DeepsortTensort  (git clone https://github.com/wang-xinyu/tensorrtx.git)
│      ├── build
│      │   └── CMakeFiles
│      │       ├── 3.10.2
│      │       │   ├── CompilerIdC
│      │       │   │   └── tmp
│      │       │   └── CompilerIdCXX
│      │       │       └── tmp
│      │       ├── CMakeTmp
│      │       ├── yololayer.dir
│      │       └── yolov3.dir
│      └── yolov3 (git clone -b archive https://github.com/ultralytics/yolov3.git)
│          ├── cfg
│          ├── data
│          │   └── samples
│          ├── __pycache__
│          ├── utils
│          │   └── __pycache__
│          └── weights
└──nabTensorRTxResource (onedrive)
   └── yolov3
       └── weights
           ├── nab_yolov3_320.cfg
           ├── nab_yolov3_320.engine
           ├── nab_yolov3_320.weights
           └── nab_yolov3_320.wts
```
## Hardware

| CPU | GPU | RAM |
| ----------- | ----------- | ----------- | 
| AMD Ryzen 7 4800H | NVIDIA Geforce GTX 1660 Ti 6GB VRAM |  16GB  |


## Inference Object Detection


| Model Object Detection | FPS_AVR (*fps*) | Memory CPU (*mB*) | Memory GPU (*miB*) | Time (*s*) |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| **YOLOv3_FP16** | 125 | 705 | 1606 | *___* |


## Inference Object Detection + Tracking

| Model | FPS_AVR (*fps*) | Memory CPU (*mB*) | Memory GPU (*miB*) | Time (*s*) |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| **YOLOv3_FP16 + Deepsort** | *___* | *___* | *___* | *___* |



## Reference
* [tensorrtx](https://github.com/wang-xinyu/tensorrtx)
* [deepsort-tensorrt](https://github.com/nabang1010/deepsort-tensorrt)
