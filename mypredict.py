import os
from ultralytics import YOLO

# 指定预训练模型的路径
model_path = r'C:\Users\kang_\PycharmProjects\pythonProject2\ultralytics-main\best.pt'
# 加载预训练模型
model = YOLO(model_path)

# 指定待预测影像文件夹路径
image_folder = r'C:\Users\kang_\Desktop\predict'

# 获取影像文件夹中的所有文件
image_files = os.listdir(image_folder)

# 循环遍历每个影像文件进行预测
for image_file in image_files:
    # 构建完整的影像文件路径
    image_path = os.path.join(image_folder, image_file)
    # 进行预测，并保存预测结果
    results = model(image_path, save=True, save_conf=True, save_txt=True, name='output')


# save=True为保存预测结果
# save_conf=True为保存坐标信息
# save_txt=True为保存txt结果，但是yolov8本身当图片中预测不到异物时，不产生txt文件