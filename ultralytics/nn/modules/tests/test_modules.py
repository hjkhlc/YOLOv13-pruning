import torch
import torch.nn as nn  # 新增：导入 nn 以支持 nn.Module()
from ultralytics.nn.modules import PruneHyperACE, compute_importance

# 测试PruneHyperACE
model = PruneHyperACE(c1=64, c2=256, num_hyperedges=8)
x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
out = model(x_list)
assert out.shape == (2, 256, 32, 32)
model.prune(0.5)  # 测试prune
assert model.branch1.m.hgnn.prune_mask.sum() < 8  # 至少剪一半

# 测试compute_importance
stats = compute_importance(model)
assert stats.shape[0] == 8 and not torch.isnan(stats).any()

# 测试边缘：无模块
try:
    stats_empty = compute_importance(nn.Module())
except ValueError as e:
    assert "No HyperACE" in str(e)

# 测试 DDP sync（模拟，非真 DDP）
model.eval()  # 确保 detach
stats = compute_importance(model, sync_dist=True)  # 警告但不崩溃
print(f"Aggregated stats shape: {stats.shape}, min/max: {stats.min():.2f}/{stats.max():.2f}")

# 测试导入兼容：原类 vs 子类
from ultralytics.nn.modules import HyperACE, PruneHyperACE
orig_model = HyperACE(c1=64, c2=256, num_hyperedges=8)
pruned_model = PruneHyperACE(c1=64, c2=256, num_hyperedges=8)
assert type(orig_model).__name__ == 'HyperACE'  # 原类无 prune
assert not hasattr(orig_model, 'prune')
assert type(pruned_model).__name__ == 'PruneHyperACE'  # 子类有 prune
assert hasattr(pruned_model, 'prune')

# 测试多模型聚合（模拟全局 prune）
models = [PruneHyperACE(64, 256), orig_model]  # 混用
multi_stats = compute_importance(models)
assert multi_stats.shape[0] == 8
print(f"Multi-model stats: {multi_stats.mean():.3f}")