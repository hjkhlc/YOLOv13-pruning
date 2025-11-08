import os
import torch  # 用于设备管理和dummy输入
from ultralytics import YOLO
from ultralytics.utils import LOGGER  # 用于自定义日志输出
try:
    from thop import profile  # Ultralytics内置thop，用于FLOPs计算
except ImportError:
    LOGGER.error("thop未安装，请运行 'pip install thop'。")
    profile = None

def compute_flops(pt_path, yaml_path='trashcan_instance.yaml', device_id=0, multi_gpu_ids=None):
    """
    加载YOLOv13 .pt模型，初始化stats后使用thop计算FLOPs（GFLOPs）和参数量（M）。
    支持单GPU（默认cuda:0）或多GPU（仅val并行，profile单GPU）。
    
    Args:
        pt_path (str): PyTorch checkpoint路径
        yaml_path (str): 数据集YAML路径，用于val初始化stats（默认trashcan_instance.yaml）
        device_id (int): 单GPU ID，默认0（cuda:0）
        multi_gpu_ids (list[int]): 多GPU ID列表，如[0,1,2,3,4,5]；若指定，则val使用多GPU
    
    Returns:
        dict: 包含'flops_g'和'params_m'的字典
    """
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"模型文件未找到: {os.path.abspath(pt_path)}")
    
    if not os.path.exists(yaml_path):
        LOGGER.warning(f"数据集YAML {yaml_path} 未找到，无法初始化stats。FLOPs计算可能不准确。")
        yaml_path = None
    
    # 设备配置：单GPU为主，multi_gpu仅用于val
    main_device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    if main_device.type == 'cuda':
        if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
            LOGGER.warning(f"主GPU {device_id} 不可用（总GPU数: {torch.cuda.device_count()}），fallback到CPU。")
            main_device = torch.device('cpu')
    LOGGER.info(f"主设备: {main_device}")
    
    # 多GPU检查
    val_device = main_device
    if multi_gpu_ids is not None:
        if torch.cuda.is_available() and len(multi_gpu_ids) <= torch.cuda.device_count():
            val_device = ','.join(map(str, multi_gpu_ids))  # e.g., '0,1,2,3,4,5'
            LOGGER.info(f"val多GPU: {val_device}")
        else:
            LOGGER.warning(f"多GPU {multi_gpu_ids} 不可用（总GPU数: {torch.cuda.device_count()}），val使用单GPU。")
            val_device = str(device_id)
    
    LOGGER.info(f"加载模型: {pt_path}")
    model = YOLO(pt_path, task='detect')  # 指定task='detect'，YOLOv13目标检测任务
    
    # 迁移模型到主设备（profile使用）
    model.to(main_device)
    
    # 使用val初始化模型stats（确保FLOPs计算完整，verbose=False仅更新不评估mAP）
    if yaml_path:
        try:
            LOGGER.info(f"初始化stats using val on {yaml_path} (device={val_device})...")
            model.val(data=yaml_path, verbose=False, batch=258, device=val_device)  # 多GPU加速初始化
            LOGGER.info("stats初始化完成")
        except Exception as e:
            LOGGER.warning(f"val初始化失败: {e}。使用默认stats计算FLOPs。")
    
    if profile is None:
        raise ImportError("thop不可用，无法计算FLOPs。请安装thop。")
    
    # 使用thop直接profile模型（单GPU，绕过info()，兼容YOLOv13 HyperACE）
    dummy_input = torch.randn(1, 3, 480, 480).to(main_device)  # dummy输入，匹配imgsz=480
    LOGGER.info("使用thop profile计算FLOPs (单GPU)...")
    flops, params = profile(model.model, inputs=(dummy_input,), verbose=False)  # 前向传播计数
    flops_g = flops / 1e9  # 转换为GFLOPs
    params_m = params / 1e6  # 参数量转换为M
    
    LOGGER.info(f"  FLOPs: {flops_g:.2f} GFLOPs")
    LOGGER.info(f"  参数量: {params_m:.2f} M")
    
    return {'flops_g': flops_g, 'params_m': params_m}

def main():
    current_dir = os.getcwd()
    baseline_pt = os.path.join(current_dir, 'baseline.pt')
    pruned_pt = os.path.join(current_dir, 'physical_pruned.pt')
    yaml_path = os.path.join(current_dir, 'trashcan_instance.yaml')  # YAML路径
    
    # 多GPU ID列表
    multi_gpu_ids = [0, 1, 2, 3, 4, 5]  # 指定GPU 0-5
    
    LOGGER.info("=== YOLOv13 FLOPs 对比计算 ===")
    LOGGER.info(f"当前目录: {current_dir}")
    
    # 计算baseline FLOPs（多GPU val）
    baseline_results = compute_flops(baseline_pt, yaml_path, device_id=0, multi_gpu_ids=multi_gpu_ids)
    
    # 计算pruned FLOPs（多GPU val）
    pruned_results = compute_flops(pruned_pt, yaml_path, device_id=0, multi_gpu_ids=multi_gpu_ids)
    
    # 输出对比
    delta_flops = pruned_results['flops_g'] - baseline_results['flops_g']
    delta_params = pruned_results['params_m'] - baseline_results['params_m']
    LOGGER.info("=== 对比结果 ===")
    LOGGER.info(f"Baseline: {baseline_results['flops_g']:.2f} GFLOPs, {baseline_results['params_m']:.2f} M")
    LOGGER.info(f"Pruned:    {pruned_results['flops_g']:.2f} GFLOPs, {pruned_results['params_m']:.2f} M")
    LOGGER.info(f"Delta:     {delta_flops:+.2f} GFLOPs ({delta_flops / baseline_results['flops_g'] * 100:+.1f}%), "
                f"{delta_params:+.2f} M ({delta_params / baseline_results['params_m'] * 100:+.1f}%)")

if __name__ == "__main__":
    main()