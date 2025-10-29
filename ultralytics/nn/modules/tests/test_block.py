import pytest
import torch
from ultralytics.nn.modules.block import HyperACE

def test_prune():
    """Test HyperACE pruning functionality."""
    model = HyperACE(c1=64, c2=256, num_hyperedges=8, channel_adjust=False)
    orig_num = model.branch1.m.hgnn.edge_generator.num_hyperedges
    x_dummy_calib = [torch.randn(1, 64, 64, 64), torch.randn(1, 64, 32, 32), torch.randn(1, 64, 16, 16)]
    with torch.no_grad():
        for i in range(3):
            out = model(x_dummy_calib)
            if i == 0:  # + 打印首次 forward 形状（调试）
                print(f"Calib forward {i+1} shapes: fuse_out={out.shape if 'out' in locals() else 'N/A'}")  # 占位，实际在 calib 后打印
    model.prune_hyperedges(0.5)
    # 检查两个分支 mask 一致
    kept1 = (model.branch1.m.hgnn.prune_mask > 0.0).sum().item()
    kept2 = (model.branch2.m.hgnn.prune_mask > 0.0).sum().item()
    assert kept1 == kept2 and kept1 < orig_num, f"Branches mismatch: branch1={kept1}, branch2={kept2} (orig={orig_num})"
    assert kept1 >= 1, "Kept at least 1 hyperedge"
    out = model(x_dummy_calib)
    assert out.shape == (1, 256, 32, 32), f"Output shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN after pruning"
    print("Prune test passed.")