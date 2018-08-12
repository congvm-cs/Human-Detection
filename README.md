# YOLO-Object-Detection optimization on Xeon scalable processors  
  
In this project, optimization of TensorFlow code is performed for an object detection application to obtain real-time performance.  
Please refer the following paper for all the details regarding performance optimizations,  
https://colfaxresearch.com/yolo-optimization/  

Rquirements:  
------------
Numpy  
Python 3 
Tensroflow   
OpenCV  


Steps to use this code:  
----------------------

1) Go to utils/ and run:   
   $ `python config.py`   
   this downloads the darknet weight files. Also, fuses batchnorm layers and creates TensorFlow Ckpt files.  
  
2) To run Webcam inference:  
   $ `python main.py`    

  
