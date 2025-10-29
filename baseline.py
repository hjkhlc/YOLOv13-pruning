from ultralytics import YOLO

model = YOLO('yolov13.yaml')

# Train the model
results = model.train(
  data='trashcan_instance.yaml',
  epochs=600, 
  batch=256, 
  imgsz=480,
  scale=0.5,  # S:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; L:0.5; X:0.6
  device="1,3,4,5",
  verbose=False
)


# Evaluate model performance on the validation set
metrics_val = model.val(data='trashcan_instance.yaml', split='val') 

# Evaluate model performance on the test set
metrics_test = model.val(data='trashcan_instance.yaml', split='test')

# # Perform object detection on an image
# results = model("path/to/your/image.jpg")
# results[0].show()