import multiprocessing
from ultralytics import YOLO

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Create a new YOLO model from scratch
   #  model = YOLO('C:/Users/kang_/PycharmProjects/pythonProject2/ultralytics-main/ultralytics/cfg/models/v8/myyolov8n.yaml')
   #
   #  # Load a pretrained YOLO model (recommended for training)
   # # model = YOLO('C:/Users/kang_/PycharmProjects/pythonProject2/ultralytics-main/yolov8n.pt')
   #
   #  # Train the model using the 'coco8.yaml' dataset for 3 epochs
   #  results = model.train(data='C:/Users/kang_/PycharmProjects/pythonProject2/ultralytics-main/ultralytics/cfg/datasets/mycoco128.yaml', epochs=3)
   #
   #  # Evaluate the model's performance on the validation set
   #  results = model.val()

    model = YOLO(r'C:\Users\kang_\PycharmProjects\pythonProject2\ultralytics-main\ultralytics\cfg\models\v8\myyolov8n.yaml')
    model.load(r'C:\Users\kang_\PycharmProjects\pythonProject2\ultralytics-main\best.pt')
    results = model.train(**{'cfg':r'C:\Users\kang_\PycharmProjects\pythonProject2\ultralytics-main\ultralytics\cfg\default.yaml', 'data':r'C:\Users\kang_\PycharmProjects\pythonProject2\ultralytics-main\data\mycoco\data.yaml'})
    # results = model.val()