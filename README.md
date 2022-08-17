# Real-Time-Car-Detection

Simple Real Time Car Detection app using feed from Youtube livestream to count vehicles.

![RTcar](https://github.com/Kropekkk/Real-Time-Car-Detection/blob/main/RTcar.gif)

## Dependencies
Simple Traffic COunter using [Yolov5](https://github.com/ultralytics/yolov5) and [SORT](https://github.com/abewley/sort) algorithm

* Python 3.9.13
* OpenCV
* PyTorch
* VidGear

## Usage

1. Create virtual environment ```python -m venv enviro```
2. Activate the virtual environment```.\enviro\Scripts\activate```
3. Install dependencies ```pip install -r requirements.txt```
4. Clone yolov5 ```git clone https://github.com/ultralytics/yolov5```
5. Run live.py using ```python live.py <URL>```
