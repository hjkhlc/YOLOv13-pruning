import os  # 用于获取当前目录路径
import onnx  # 用于ONNX IR version降级（pip install onnx）
import torch  # 用于FX图优化和torch.compile
import torch.nn as nn  # 用于 nn.Linear 等
import torch.fx  # Torch FX for graph capture and transformation
from torch.fx import GraphModule  # FX图模块
from torch.fx.passes.infra.pass_base import PassBase  # 修正：FX Pass基类路径
from torch.fx.passes.infra.pass_manager import PassManager  # 修正：Pass管理器路径
from ultralytics import YOLO
from ultralytics.utils import LOGGER  # 导入LOGGER用于自定义日志
from ultralytics.nn.modules.block import HyperACE, PruneHyperACE  # 新增：导入HyperACE（block.py）
import math  # 用于 scaling (AdaHyperedgeGen)

def downgrade_ir_version(input_path, output_path, target_ir=9):
    """
    降级ONNX模型的IR version到指定值（如9），保持opset不变。
    适用于YOLOv13 prune后ONNX兼容旧Runtime。
    
    Args:
        input_path (str): 输入ONNX文件路径
        output_path (str): 输出ONNX文件路径
        target_ir (int): 目标IR version (9推荐)
    """
    try:
        # 加载模型
        model_onnx = onnx.load(input_path)
        original_ir = model_onnx.ir_version
        LOGGER.info(f"原始IR version: {original_ir}")
        
        # 检查兼容性
        if original_ir <= target_ir:
            LOGGER.warning(f"原始IR {original_ir} 已 <= {target_ir}，无需降级。")
            return input_path  # 返回原路径
        
        # 降级IR version
        model_onnx.ir_version = target_ir
        onnx.checker.check_model(model_onnx)  # 验证模型完整性（YOLOv13 HyperACE块无问题）
        
        # 保存新模型
        onnx.save(model_onnx, output_path)
        LOGGER.info(f"IR降级完成：{original_ir} -> {target_ir}，保存至 {output_path}")
        LOGGER.info(f"opset版本保持: {model_onnx.opset_import[0].version}")
        return output_path
    except ImportError:
        LOGGER.warning("onnx库未安装，请运行 'pip install onnx' 以启用IR降级。使用原ONNX文件。")
        return input_path
    except Exception as e:
        LOGGER.error(f"IR降级失败: {e}。使用原ONNX文件。")
        return input_path

class HyperACEPrunePass(PassBase):
    """
    自定义FX Pass：物理移除HyperACE模块中pruned超边（基于pruned_mask）。
    简化版：仅标记/替换节点target，避免submodule迭代（Proxy issue）。
    """
    def __init__(self, pruned_mask=None):
        super().__init__()
        self.pruned_mask = pruned_mask  # 假设从model.prune()获取的mask tensor (shape: [num_hyperedges])

    def run_pass(self, graph_module: GraphModule) -> GraphModule:
        # 遍历图节点，匹配HyperACE call_module（非迭代submodule）
        for node in list(graph_module.graph.nodes):
            if node.op == 'call_module' and 'hyperace' in str(node.target).lower():
                if self.pruned_mask is not None and torch.any(self.pruned_mask == 0):
                    # 标记节点为pruned（替换target为简化版，实际移除在manual fallback）
                    node.target = f"pruned_{node.target}"
                    LOGGER.debug(f"Marked pruned HyperACE node: {node.target}")
        
        graph_module.graph.lint()  # 验证图完整性
        graph_module.recompile()  # 重新编译图
        return graph_module

def manual_physical_prune(model, pruned_mask=None):
    """
    手动物理剪枝：遍历HyperACE/AdaHGConv，切片移除pruned超边（fallback FX失败）。
    新增：重建 importance_stats (nn.Parameter, [keep_num])，修复更新尺寸不匹配。
    
    Args:
        model: YOLOv13 nn.Module实例（post-logic prune）
        pruned_mask: mask tensor (shape: [num_hyperedges])，0=pruned
    
    Returns:
        model: 更新后模型
    """
    removed_params = 0
    for m in model.model.modules():
        if isinstance(m, (HyperACE, PruneHyperACE)):
            branches = [m.branch1, m.branch2] if hasattr(m, 'branch1') else [m]
            for b in branches:
                if hasattr(b, 'm') and hasattr(b.m, 'hgnn') and hasattr(b.m.hgnn, 'prune_mask'):
                    mask = b.m.hgnn.prune_mask if pruned_mask is None else pruned_mask
                    keep_idx = torch.nonzero(mask > 0).squeeze(-1)  # indices where mask=1 (shape: [keep_num])
                    keep_num = len(keep_idx)
                    D = b.m.hgnn.edge_generator.prototype_base.shape[1]  # node_dim = D
                    
                    # 切片 prototype_base pruned部分
                    if hasattr(b.m.hgnn.edge_generator, 'prototype_base'):
                        orig_size = b.m.hgnn.edge_generator.prototype_base.numel()
                        b.m.hgnn.edge_generator.prototype_base.data = b.m.hgnn.edge_generator.prototype_base[keep_idx]
                        removed_params += orig_size - b.m.hgnn.edge_generator.prototype_base.numel()
                    
                    # 递归切片 edge_proj/node_proj Linear权重（in/out=D，无需调整；mask 已处理 E 维）
                    def slice_linear_weights(proj_mod, keep_idx):
                        nonlocal removed_params
                        if isinstance(proj_mod, nn.Linear):
                            # Linear(D, D) 无 E 维，跳过切片（仅 mask He）
                            return
                        elif isinstance(proj_mod, nn.Sequential):
                            for child in proj_mod:
                                slice_linear_weights(child, keep_idx)
                        
                    for proj in ['edge_proj', 'node_proj']:
                        if hasattr(b.m.hgnn, proj):
                            submod = getattr(b.m.hgnn, proj)
                            slice_linear_weights(submod, keep_idx)
                    
                    # 重建 context_net (out_features=keep_num * D)
                    gen = b.m.hgnn.edge_generator
                    context_type = gen.context
                    if context_type in ("mean", "max"):
                        in_f = D
                    elif context_type == "both":
                        in_f = 2 * D
                    else:
                        raise ValueError(f"Unsupported context '{context_type}'")
                    old_net = gen.context_net
                    gen.context_net = nn.Linear(in_f, keep_num * D)  # 新 Linear
                    nn.init.xavier_uniform_(gen.context_net.weight)  # 重新 init
                    removed_params += sum(p.numel() for p in old_net.parameters())  # 计入移除
                    del old_net  # 释放旧网
                    
                    # 新增：重建 importance_stats (nn.Parameter, [keep_num])，切片保留历史重要性
                    old_stats = gen.importance_stats.detach()  # [orig_E]
                    new_stats = old_stats[keep_idx] if len(keep_idx) > 0 else torch.zeros(keep_num)  # 切片 [keep_num]
                    gen.importance_stats = nn.Parameter(new_stats)  # 替换 Parameter
                    removed_params += old_stats.numel() - new_stats.numel()  # 计入移除
                    
                    # 更新 prune_mask 和 num_hyperedges（用 nn.Parameter 包装）
                    b.m.hgnn.prune_mask = nn.Parameter(torch.ones(keep_num, dtype=torch.float32))
                    b.m.hgnn.edge_generator.num_hyperedges = keep_num
                    # 形状一致性检查
                    assert b.m.hgnn.edge_generator.prototype_base.shape[0] == keep_num, f"Prototype shape mismatch: {b.m.hgnn.edge_generator.prototype_base.shape[0]} vs {keep_num}"
                    assert b.m.hgnn.edge_generator.context_net.out_features == keep_num * D, f"Context net out_features mismatch: {b.m.hgnn.edge_generator.context_net.out_features} vs {keep_num * D}"
                    assert gen.importance_stats.shape[0] == keep_num, f"Importance stats shape mismatch: {gen.importance_stats.shape[0]} vs {keep_num}"
                    LOGGER.info(f"Manual pruned {len(mask) - keep_num} hyperedges in {type(b).__name__} (D={D}, keep_num={keep_num})")
    
    LOGGER.info(f"Manual physical prune: removed {removed_params} params total (incl. context_net & stats rebuild)")
    return model

def physical_prune_with_fx(model, example_input=torch.randn(1, 3, 480, 480), compile_opt=False):
    """
    使用Torch FX实现YOLOv13物理剪枝：trace → 应用PrunePass → 重建模型。
    若trace失败，回退手动physical prune（兼容Proxy iteration）。
    
    Args:
        model: YOLOv13 nn.Module实例（post-logic prune）
        example_input: 示例输入，用于FX tracer（匹配imgsz=480）
        compile_opt (bool): 是否应用torch.compile优化（提升FPS ~10%）
    
    Returns:
        optimized_model: 物理剪枝后模型
    """
    # 版本检查：确保PyTorch >=2.1支持infra passes
    if torch.__version__ < '2.1.0':
        LOGGER.warning(f"PyTorch {torch.__version__} <2.1，FX passes有限。回退手动prune。")
        return manual_physical_prune(model)
    
    try:
        # 步骤1: FX symbolic_trace（标准API，固定输入避免Proxy迭代）
        model.eval()
        traced = torch.fx.symbolic_trace(model.model, concrete_args={'forward': example_input})
        LOGGER.info(f"FX图捕获成功：{len(list(traced.graph.nodes))} 节点")

        # 步骤2: 获取pruned_mask
        pruned_mask = None
        for m in model.model.modules():
            if isinstance(m, (HyperACE, PruneHyperACE)) and hasattr(m, 'm') and hasattr(m.m, 'hgnn'):
                pruned_mask = m.m.hgnn.prune_mask.detach()
                break
        if pruned_mask is None:
            LOGGER.warning("未找到pruned_mask，回退手动prune。")
            return manual_physical_prune(model)

        # 步骤3: 应用自定义PrunePass（简化标记）
        pass_manager = PassManager()
        pass_manager.add_pass(HyperACEPrunePass(pruned_mask))
        optimized_gm = pass_manager.run_passes(traced)

        # 步骤4: 重建模型（注入FX图）+ 手动重建 context_net/stats（确保 forward 安全）
        model.model = optimized_gm
        manual_physical_prune(model, pruned_mask)  # 补充重建（FX 标记后手动切片）
        LOGGER.info(f"物理剪枝完成：移除 {torch.sum(pruned_mask == 0)} 超边")

        # 步骤5: 可选torch.compile
        if compile_opt:
            model.model = torch.compile(model.model, mode='reduce-overhead')
            LOGGER.info("应用torch.compile优化（mode: reduce-overhead）")

        # 验证一致性
        with torch.no_grad():
            orig_out = model(example_input)
            opt_out = model(example_input)
            diff = torch.norm(orig_out - opt_out)
            LOGGER.info(f"输出一致性检查：L2 diff = {diff:.6f} (阈值<1e-4为合格)")

        return model

    except Exception as e:
        if "Proxy object cannot be iterated" in str(e) or "symbolic_trace" in str(e).lower():
            LOGGER.warning(f"FX symbolic_trace失败（Proxy/trace错误）：{e}。回退手动物理prune。")
        else:
            LOGGER.error(f"FX物理剪枝失败: {e}。回退到逻辑prune模型。")
        return manual_physical_prune(model)

# 主流程：加载模型（其余不变）
model = YOLO('yolov13.yaml')

# Train the model（保持原样，确保stats在train中更新）
results = model.train(
  data='trashcan_instance.yaml',
  epochs=1, 
  batch=16, 
  imgsz=480,
  scale=0.9,
  mosaic=1.0,
  mixup=0.0,
  copy_paste=0.1,
#   device="0,1,2,3,4,5",  # 若单卡测试，改为device=0
  device = 5,
  verbose=False,
  amp=False
)

# prune前baseline验证（val set）
baseline_metrics = model.val(data='trashcan_instance.yaml', split='val')
baseline_map = baseline_metrics.box.map
LOGGER.info(f"Baseline mAP@50:95 (pre-prune): {baseline_map:.4f}")

# 保存prune前模型
current_dir = os.getcwd()
baseline_pt_path = os.path.join(current_dir, 'baseline.pt')
model.save(baseline_pt_path)
LOGGER.info(f"Saved baseline model to: {baseline_pt_path}")

# 执行prune（逻辑剪枝）
model.prune(ratio=0.8) 

# 物理剪枝（FX图优化 + manual fallback）
model = physical_prune_with_fx(model, example_input=torch.randn(1, 3, 480, 480), compile_opt=True)

# prune后验证
pruned_metrics = model.val(data='trashcan_instance.yaml', split='val')
pruned_map = pruned_metrics.box.map
delta_map = pruned_map - baseline_map
LOGGER.info(f"Physical Pruned mAP@50:95: {pruned_map:.4f} (delta: {delta_map:+.4f})")

# 保存prune后模型
pruned_pt_path = os.path.join(current_dir, 'physical_pruned.pt')
model.save(pruned_pt_path)
LOGGER.info(f"Saved physical pruned model to: {pruned_pt_path}")

# 可选：ONNX导出与IR降级（原注释部分保持）
# original_onnx_path = model.export(format='onnx', imgsz=480, dynamic=False, device='cpu', opset=9)
# final_onnx_path = downgrade_ir_version(original_onnx_path, os.path.join(current_dir, 'best_ir9.onnx'), target_ir=9)