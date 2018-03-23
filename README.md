# py_opencv_mobilenet_practice

# Setup OpenCV3.3 or later

## Ubuntu
```
sudo apt install python3-opencv
```

## Raspbian
```
sudo apt install pip3
sudo pip3 install opencv_contrib_python
```

# Usage
```
python3 mobilenet_scan_camera.py
```

If you don't need preview window.
```
python3 mobilenet_scan_camera.py --showpreview=false
```

# References
- [shicai/MobileNet-Caffe](https://github.com/shicai/MobileNet-Caffe)
- [【Windows】【Python】OpenCV3.3.1のdnnモジュールサンプル(mobilenet_ssd_python.py)](https://qiita.com/Kazuhito/items/e2b57db762b183238b13)
- [opencv/samples/dnn/mobilenet_ssd_accuracy.py](https://github.com/opencv/opencv/blob/master/samples/dnn/mobilenet_ssd_accuracy.py)
- [Deep Neural Networks (dnn module) | OpenCV 3.3](https://docs.opencv.org/3.3.0/d2/d58/tutorial_table_of_content_dnn.html)
- [Failed to load caffe model in opencv 3.3](https://github.com/opencv/opencv/issues/9651)
- [opencv3.3.0 with deep learning](https://hackmd.io/s/S1gWq7BwW)
- [models/research/slim](https://github.com/tensorflow/models/tree/376dc8dd0999e6333514bcb8a6beef2b5b1bb8da/research/slim)
