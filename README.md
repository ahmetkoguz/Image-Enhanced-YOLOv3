# Image-Enhanced-YOLOv3
* Demo video: https://1drv.ms/v/c/64907ae47798c14c/EVu5Bcr2vuFPjpSzx_1_NHIBq3s0i8BM_LPGkz5eBX9yZA?e=5Tm8s6

* To get started, you need to first download the YOLOv3 weights. To do so, run the following command in the yolo-coco folder: `wget https://pjreddie.com/media/files/yolov3.weights`.

* To download the dataset used (RESIDE), go to the following website: https://utexas.app.box.com/s/2yekra41udg9rgyzi3ysi513cps621qz. Once downloaded and unzipped, under Dataset/VOCdevkit/VOC2007, place the Annotations and JPEGImages folders, replacing the current ones. As an example, one image and one annotation is placed to show the necessary format.

* When loading in the data, the VOCDetection loader function in image_enhanced_yolov3 wants images in jpg but dataset was provided in png, so either convert images or change function to use png in the VOCBase class.

* To download dependencies, run: `pip install -r ./docs/requirements.txt`.

* To run enhanced model: `python image_enhanced_yolov3.py`. To run YOLOv3 model, without any filtering `python yolov3_detection.py`

* To run the enhanced model on the overall test dataset or just see the result on one image, adjust the show_images parameter in prediction_accuracy function.
