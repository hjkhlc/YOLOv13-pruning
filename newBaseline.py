from ultralytics import YOLO
from ultralytics.utils import LOGGER  # 新增：导入LOGGER用于自定义日志

model = YOLO('yolov13.yaml')

# Train the model（保持原样，确保stats在train中更新）
results = model.train(
  data='trashcan_instance.yaml',
  epochs=3, 
  batch=256, 
  imgsz=480,
  scale=0.9,  # 修改：统一为0.9（原注释S/L/X均为0.9，避免混淆）
  mosaic=1.0,
  mixup=0.0,  # S:0.05; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; L:0.5; X:0.6
  device="1,3,4,5",  # 若单卡测试，改为device=0
  verbose=False,
  amp=False  # 新增：禁用自动混合精度，避免Half/Float冲突
)

# # 新增：prune前baseline验证（val set）
# baseline_metrics = model.val(data='trashcan_instance.yaml', split='val')
# baseline_map = baseline_metrics.box.map  # mAP@50:95
# LOGGER.info(f"Baseline mAP@50:95 (pre-prune): {baseline_map:.4f}")

# 新增：执行prune（ratio=0.2，剪枝20%超边）
model.prune(ratio=0.2)  # 调用任务13方法，日志会输出"Pruned X HyperACE modules with ratio=0.20"

# # 新增：prune后验证（val set），计算delta
# pruned_metrics = model.val(data='trashcan_instance.yaml', split='val')
# pruned_map = pruned_metrics.box.map  # mAP@50:95
# delta_map = pruned_map - baseline_map
# LOGGER.info(f"Pruned mAP@50:95: {pruned_map:.4f} (delta: {delta_map:+.4f})")

# # 原：test set验证（prune后运行）
# metrics_test = model.val(data='trashcan_instance.yaml', split='test')
# test_map = metrics_test.box.map
# LOGGER.info(f"Test mAP@50:95 (post-prune): {test_map:.4f}")

# # Perform object detection on an image（可选，测试推理）
# results = model("path/to/your/image.jpg")
# results[0].show()