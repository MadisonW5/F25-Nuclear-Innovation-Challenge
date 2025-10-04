#instructions to enable GPU based training
from ultralytics import YOLO
import cv2



#path to your data.yaml file
training_set = r"C:\Users\snail\Downloads\My First Project.v3i.yolov11\data.yaml"


def main():

    model = YOLO("yolo11n.pt")
    results = model.train(data = training_set,
                          epochs=10, 
                          imgsz=640, 
                          batch=8,#can be modified to lower or higher to decrease/increase VRAM usage
                          #device="mps" use if your device is a Mac
                          device = "cpu", #use if your device don't have GPU, warning: it's significantly slower
                          #device = 0,
                        workers=4 ) #experiment with 2-4 as a start, feel free to modify if your GPU/CPU is capable of handling more threads
    #tip for batch and workers selection: 
    #open task manager to check GPU usage after starting training. If GPU usage hovers around 80-100% most of time, then this is a sweet spot.
    #if GPU usage is lower than this, try increasing workers
    #if memory usage is low, try increasing batch number

if __name__ == "__main__":
    main()

