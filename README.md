making the correct virtual environment:
1) type: cd [insert absolute path to your project locally]
2) type: py -3.12 -m venv .venv
3) type: .\.venv\Scripts\activate to activate your virtual environment
4) type: python -m pip install --upgrade pip
5) Visit Pytorch: https://pytorch.org/get-started/locally/ and match the settings to your computer (choose CPU if you don't have a dedicated GPU) (I personally typed: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126)
6) type: pip install ultralytics opencv-python

note: create a folder called "dataset photos" to add your photos to train

image_capturer.py = takes photos used for training
model was made with the photos on Roboflow
yolo11-custom.py = train model
main.py = use model with your camera
