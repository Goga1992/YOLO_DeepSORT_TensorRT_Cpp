# YOLO_DeepSORT_TensorRT_Cpp


# yolov3_NAB


## Hardware

| CPU | GPU | RAM |
| ----------- | ----------- | ----------- | 
| AMD Ryzen 7 4800H | NVIDIA Geforce GTX 1660 Ti 6GB VRAM |  16GB  |


## Inference Object Detection (C++ API)



| Model Object Detection | FPS_AVR (*fps*) | Memory CPU (*mB*) | Memory GPU (*miB*) | Time (*s*) |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| **YOLOv3_FP16** | 125 | 705 | 1606 | *___* |


## Inference Object Detection + Tracking (C++ API)

| Model | FPS_AVR (*fps*) | Memory CPU (*mB*) | Memory GPU (*miB*) | Time (*s*) |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| **YOLOv5_FP16 + DeepSORT** | 250 | *___* | *___* | *___* |
| **YOLOv4_FP16 + DeepSORT** | *___* | *___* | *___* | *___* |
| **YOLOv3_FP16 + DeepSORT** | 166.667 | 1601 | 971 | *___* |



## Reference
* [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)
* [RichardoMrMu/deepsort-tensorrt](https://github.com/RichardoMrMu/deepsort-tensorrt)
* [RichardoMrMu/yolov5-deepsort-tensorrt](https://github.com/RichardoMrMu/yolov5-deepsort-tensorrt)
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [ultralytics/yolov3](https://github.com/ultralytics/yolov3)
