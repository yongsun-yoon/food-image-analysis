# food-image-analysis


## Run
```
git clone https://github.com/heartexlabs/label-studio-ml-backend
cd label-studio-ml-backend/
pip install -U -e .
cd ..

label-studio-ml init ml_backend --script label_studio_ml_backend.py --force
label-studio-ml start ml_backend
```


```
python train.py --workers 8 --device 0 --batch-size 32 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml
python detect.py --weights food.pt --conf 0.25 --img-size 640 --source inference/images/food_001.png
```