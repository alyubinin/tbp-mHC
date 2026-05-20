"""
Comprehensive Tests for Hyper-Connection Module

This module provides tests to verify:
1. API compatibility across all variants
2. Inheritance hierarchy correctness
3. Forward/backward pass functionality
4. Double stochasticity properties (where applicable)
5. Parameter counting and shapes
6. ORTBP2N optimizer grouping and log_stats (Phases 5–6)

Run tests:
    python -m hyper_conn.tests

Or import and run specific tests:
    from hyper_conn.tests import run_all_tests
    run_all_tests()
"""

import os
import sys
import torch
from torch import nn
import numpy as np


def _repo_root_on_path():
    """Allow `import model` when running `python -m hyper_conn.tests` from site-packages layouts."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    if root not in sys.path:
        sys.path.insert(0, root)


def test_ortbp_optimizer_groups():
    """ORTBP residual-chart params land in dedicated optimizer groups when overrides are passed."""
    _repo_root_on_path()
    from model import GPT, GPTConfig

    cfg = GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_embd=8,
        hyper_conn_n=4,
        hyper_conn_type="ortbp2n_mhc",
        ortbp_log_stats=False,
    )
    model = GPT(cfg)
    overrides = {
        "ortbp_residual_chart": {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "betas": (0.8, 0.95),
        },
        "ortbp_residual_scale": {
            "lr": 3e-5,
            "weight_decay": 0.0,
            "betas": (0.8, 0.95),
        },
        "ortbp_delta": {
            "lr": 2e-5,
            "weight_decay": 0.0,
            "betas": (0.8, 0.95),
        },
    }
    opt = model.configure_optimizers(
        0.1, 6e-4, (0.9, 0.95), "cpu", module_param_group_overrides=overrides
    )
    named = dict(model.named_parameters())
    param_to_group = {}
    for g in opt.param_groups:
        gn = g.get("group_name", "")
        for p in g["params"]:
            for pname, q in named.items():
                if id(q) == id(p):
                    param_to_group[pname] = gn
                    break

    chart_names = [
        n
        for n in param_to_group
        if "dynamic_res_alpha_fn" in n or "static_alpha_res" in n
    ]
    assert chart_names, "expected ORTBP chart params in model"
    assert all(param_to_group[n] == "ortbp_residual_chart" for n in chart_names)

    scale_names = [n for n in param_to_group if n.endswith("residual_scale")]
    assert scale_names
    assert all(param_to_group[n] == "ortbp_residual_scale" for n in scale_names)

    delta_names = [n for n in param_to_group if "delta_logit" in n]
    assert delta_names
    assert all(param_to_group[n] == "ortbp_delta" for n in delta_names)

    wte = param_to_group.get("transformer.wte.weight")
    assert wte in ("default_decay", "default_nodecay")

    print("  ORTBP optimizer groups: PASS")
    return True


def test_ortbp_optimizer_no_overrides_two_groups():
    """Without module_param_group_overrides, ORTBP uses the same decay/nodecay split as other variants."""
    _repo_root_on_path()
    from model import GPT, GPTConfig

    cfg = GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_embd=8,
        hyper_conn_n=4,
        hyper_conn_type="ortbp2n_mhc",
    )
    model = GPT(cfg)
    opt = model.configure_optimizers(
        0.1, 1e-3, (0.9, 0.95), "cpu", module_param_group_overrides=None
    )
    group_names = {g.get("group_name") for g in opt.param_groups}
    assert group_names == {"default_decay", "default_nodecay"}

    print("  ORTBP optimizer (no overrides, two groups): PASS")
    return True


def test_non_ortbp_optimizer_no_ortbp_groups():
    """Non-ORTBP models must not create ortbp_* optimizer groups."""
    _repo_root_on_path()
    from model import GPT, GPTConfig

    cfg = GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_embd=8,
        hyper_conn_n=4,
        hyper_conn_type="kromhc",
    )
    model = GPT(cfg)
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    group_names = [g.get("group_name") for g in opt.param_groups]
    assert "ortbp_residual_chart" not in group_names
    assert "ortbp_residual_scale" not in group_names
    assert "ortbp_delta" not in group_names

    print("  Non-ORTBP optimizer (no ortbp_* groups): PASS")
    return True


def test_ortbp_log_stats_smoke():
    """ORTBP log_stats records scalars after forward; backward still runs."""
    _repo_root_on_path()
    from model import GPT, GPTConfig

    cfg = GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_embd=8,
        hyper_conn_n=4,
        hyper_conn_type="ortbp2n_mhc",
        ortbp_log_stats=True,
    )
    model = GPT(cfg)
    idx = torch.randint(0, 64, (2, 8))
    logits, loss = model(idx, idx)
    loss.backward()

    saw = False
    for module in model.modules():
        if hasattr(module, "get_stats"):
            stats = module.get_stats()
            if stats and "residual_scale" in stats:
                saw = True
                break
    assert saw, "expected ORTBP get_stats() after forward with ortbp_log_stats=True"

    print("  ORTBP log_stats smoke: PASS")
    return True


def test_api_compatibility():
    """Test that all variants work through the unified interface."""
    from hyper_conn import hyper_conn_init_func
    
    variants = ['none', 'hc', 'mhc', 'mhc_lite', 'kromhc', 'tbp_mhc', 'rtbp_mhc', 'srtbp_mhc', 'rtbp2n_mhc', 'srtbp2n_mhc', 'ortbp2n_mhc', 'msrtbp2n_mhc', 'amsrtbp2n_mhc']
    results = {}
    
    print("Testing API compatibility...")
    for variant in variants:
        try:
            # Get init function and expand/reduce
            init_hc, expand, reduce = hyper_conn_init_func(variant, 4)
            
            # Create layer with branch
            branch = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 512))
            layer = init_hc(dim=512, branch=branch)
            
            # Test forward pass
            x = torch.randn(2, 10, 512)
            x_expanded = expand(x)
            y = layer(x_expanded)
            z = reduce(y)
            
            assert z.shape == (2, 10, 512), f"Expected (2, 10, 512), got {z.shape}"
            results[variant] = "PASS"
            print(f"  {variant}: PASS")
        except Exception as e:
            results[variant] = f"FAIL: {e}"
            print(f"  {variant}: FAIL - {e}")
    
    return all(r == "PASS" for r in results.values())


def test_inheritance_hierarchy():
    """Test that all variants inherit from the correct base classes."""
    from hyper_conn.base import BaseHyperConnections
    from hyper_conn.mhc import ManifoldConstrainedHyperConnections
    from hyper_conn.mhc_lite import MHCLite
    from hyper_conn.tbp_mhc import TBP_MHC
    from hyper_conn.rtbpHC import RTBP_MHC
    from hyper_conn.srtbpHC import SRTBP_MHC
    from hyper_conn.rtbp2n_HC import RTBP2N_MHC
    from hyper_conn.srtbp2n_mhc import SRTBP2N_MHC
    from hyper_conn.ortbp2n_mhc import ORTBP2N_MHC
    from hyper_conn.msrtbp2n_mhc import MSRTBP2N_MHC
    from hyper_conn.amsrtbp2n_mhc import AMSRTBP2N_MHC
    from hyper_conn.Kromhc import KromHC
    from hyper_conn.hyper_connections import HyperConnections
    from hyper_conn.mhc_analysis import MHCAnalysis
    
    print("\nTesting inheritance hierarchy...")
    
    # Test BaseHyperConnections inheritance
    assert issubclass(ManifoldConstrainedHyperConnections, BaseHyperConnections)
    assert issubclass(MHCLite, BaseHyperConnections)
    assert issubclass(TBP_MHC, BaseHyperConnections)
    assert issubclass(RTBP_MHC, BaseHyperConnections)
    assert issubclass(SRTBP_MHC, BaseHyperConnections)
    assert issubclass(RTBP2N_MHC, BaseHyperConnections)
    assert issubclass(SRTBP2N_MHC, BaseHyperConnections)
    assert issubclass(ORTBP2N_MHC, BaseHyperConnections)
    assert issubclass(MSRTBP2N_MHC, BaseHyperConnections)
    assert issubclass(AMSRTBP2N_MHC, BaseHyperConnections)
    assert issubclass(KromHC, BaseHyperConnections)
    assert issubclass(HyperConnections, BaseHyperConnections)
    print("  All variants inherit from BaseHyperConnections: PASS")
    
    # Test MHCAnalysis inheritance
    assert issubclass(MHCAnalysis, ManifoldConstrainedHyperConnections)
    print("  MHCAnalysis inherits from ManifoldConstrainedHyperConnections: PASS")
    
    return True


def test_forward_backward():
    """Test forward and backward passes work correctly."""
    from hyper_conn import hyper_conn_init_func
    
    print("\nTesting forward/backward passes...")
    
    variants = ['hc', 'mhc', 'mhc_lite', 'kromhc', 'tbp_mhc', 'rtbp_mhc', 'srtbp_mhc', 'rtbp2n_mhc', 'srtbp2n_mhc', 'ortbp2n_mhc', 'msrtbp2n_mhc', 'amsrtbp2n_mhc']
    
    for variant in variants:
        init_hc, expand, reduce = hyper_conn_init_func(variant, 4)
        
        branch = nn.Linear(512, 512)
        layer = init_hc(dim=512, branch=branch)
        
        # Forward
        x = torch.randn(8, 10, 512, requires_grad=True)
        y = layer(x)
        
        # Backward
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None, f"{variant}: Input gradient should exist"
        assert not torch.isnan(x.grad).any(), f"{variant}: Gradient contains NaN"
        print(f"  {variant}: PASS")
    
    return True


def test_functional_api():
    """Test the functional (no-branch) API."""
    from hyper_conn import KromHC, MHCLite, TBP_MHC, RTBP_MHC, SRTBP_MHC, RTBP2N_MHC, SRTBP2N_MHC, ORTBP2N_MHC, MSRTBP2N_MHC, AMSRTBP2N_MHC
    
    print("\nTesting functional API (no branch)...")
    
    classes = [KromHC, MHCLite, TBP_MHC, RTBP_MHC, SRTBP_MHC, RTBP2N_MHC, SRTBP2N_MHC, ORTBP2N_MHC, MSRTBP2N_MHC, AMSRTBP2N_MHC]
    
    for cls in classes:
        layer = cls(4, dim=512)  # No branch
        
        x = torch.randn(8, 10, 512)
        branch_input, add_residual = layer(x)
        
        # User applies their own branch
        branch_output = branch_input * 2  # Simple "branch"
        
        output = add_residual(branch_output)
        
        assert output.shape == x.shape, f"{cls.__name__}: Shape mismatch"
        print(f"  {cls.__name__}: PASS")
    
    return True


def test_decorate_branch():
    """Test the decorator pattern API."""
    from hyper_conn import KromHC
    
    print("\nTesting decorate_branch API...")
    
    layer = KromHC(4, dim=512)
    
    @layer.decorate_branch
    def my_branch(x):
        return x * 2
    
    x = torch.randn(8, 10, 512)
    output = my_branch(x)
    
    assert output.shape == x.shape, "Output shape should match input"
    print("  decorate_branch: PASS")
    
    return True


def test_double_stochasticity():
    """Test that exact DS variants produce doubly stochastic matrices."""
    torch.manual_seed(0)
    from hyper_conn import KromHC, MHCLite, TBP_MHC, RTBP_MHC, SRTBP_MHC, RTBP2N_MHC, SRTBP2N_MHC, ORTBP2N_MHC, MSRTBP2N_MHC, AMSRTBP2N_MHC
    from hyper_conn.rtbpHC import recursive_transport_birkhoff
    from hyper_conn.srtbpHC import scaled_recursive_transport_birkhoff
    from hyper_conn.rtbp2n_HC import recursive_transport_birkhoff_power2
    from hyper_conn.srtbp2n_mhc import scaled_recursive_transport_birkhoff_power2
    from hyper_conn.msrtbp2n_mhc import margined_scaled_recursive_transport_birkhoff_power2
    
    print("\nTesting double stochasticity (exact DS variants)...")
    
    # These variants should produce EXACT DS matrices
    exact_ds_classes = [KromHC, MHCLite, TBP_MHC, RTBP_MHC, SRTBP_MHC, RTBP2N_MHC, SRTBP2N_MHC, ORTBP2N_MHC, MSRTBP2N_MHC, AMSRTBP2N_MHC]
    
    for cls in exact_ds_classes:
        layer = cls(4, dim=512)
        
        x = torch.randn(8, 10, 512)
        
        # Access the width_connection to get the mixing matrices
        branch_input, residuals, kwargs = layer.width_connection(x)
        
        # For now, just verify the forward pass completes
        # Full DS verification would require accessing internal alpha matrix
        print(f"  {cls.__name__}: Forward pass OK")

    # Directly verify the recursive mapper on both even and odd sizes.
    for n in (4, 5):
        params = torch.randn(3, n - 1, n - 1)
        ds = recursive_transport_birkhoff(params)
        row_sums = ds.sum(dim=-1)
        col_sums = ds.sum(dim=-2)
        ones = torch.ones_like(row_sums)
        assert torch.allclose(row_sums, ones, atol=1e-5), f"n={n}: row sums not 1"
        assert torch.allclose(col_sums, ones, atol=1e-5), f"n={n}: col sums not 1"
        assert ds.min() >= -1e-6, f"n={n}: matrix has negative entries (min={ds.min().item()})"
        print(f"  recursive_transport_birkhoff(n={n}): Exact DS OK")

    for n in (4, 5):
        params = torch.randn(3, n - 1, n - 1)
        ds = scaled_recursive_transport_birkhoff(params)
        row_sums = ds.sum(dim=-1)
        col_sums = ds.sum(dim=-2)
        ones = torch.ones_like(row_sums)
        assert torch.allclose(row_sums, ones, atol=1e-5), f"n={n}: row sums not 1"
        assert torch.allclose(col_sums, ones, atol=1e-5), f"n={n}: col sums not 1"
        assert ds.min() >= -1e-6, f"n={n}: matrix has negative entries (min={ds.min().item()})"
        print(f"  scaled_recursive_transport_birkhoff(n={n}): Exact DS OK")

    for n in (4, 8):
        params = torch.randn(3, n - 1, n - 1)
        ds = recursive_transport_birkhoff_power2(params)
        row_sums = ds.sum(dim=-1)
        col_sums = ds.sum(dim=-2)
        ones = torch.ones_like(row_sums)
        assert torch.allclose(row_sums, ones, atol=1e-5), f"n={n}: row sums not 1"
        assert torch.allclose(col_sums, ones, atol=1e-5), f"n={n}: col sums not 1"
        assert ds.min() >= -1e-6, f"n={n}: matrix has negative entries (min={ds.min().item()})"
        print(f"  recursive_transport_birkhoff_power2(n={n}): Exact DS OK")

    for n in (4, 8):
        params = torch.randn(3, n - 1, n - 1)
        ds = scaled_recursive_transport_birkhoff_power2(params)
        row_sums = ds.sum(dim=-1)
        col_sums = ds.sum(dim=-2)
        ones = torch.ones_like(row_sums)
        assert torch.allclose(row_sums, ones, atol=1e-5), f"n={n}: row sums not 1"
        assert torch.allclose(col_sums, ones, atol=1e-5), f"n={n}: col sums not 1"
        assert ds.min() >= -1e-6, f"n={n}: matrix has negative entries (min={ds.min().item()})"
        print(f"  scaled_recursive_transport_birkhoff_power2(n={n}): Exact DS OK")

    for n in (4, 8):
        params = torch.randn(3, n - 1, n - 1)
        ds = margined_scaled_recursive_transport_birkhoff_power2(params)
        row_sums = ds.sum(dim=-1)
        col_sums = ds.sum(dim=-2)
        ones = torch.ones_like(row_sums)
        assert torch.allclose(row_sums, ones, atol=1e-5), f"n={n}: row sums not 1"
        assert torch.allclose(col_sums, ones, atol=1e-5), f"n={n}: col sums not 1"
        assert ds.min() >= -1e-6, f"n={n}: matrix has negative entries (min={ds.min().item()})"
        print(f"  margined_scaled_recursive_transport_birkhoff_power2(n={n}): Exact DS OK")
    
    return True


def test_mhc_analysis_logging():
    """Test MHCAnalysis logging functionality."""
    from hyper_conn import MHCAnalysis
    
    print("\nTesting MHCAnalysis logging...")
    
    layer = MHCAnalysis(4, dim=512)
    layer.log_info = True
    
    x = torch.randn(8, 10, 512)
    y = layer(x)
    
    assert 'H_res_bef' in layer.info, "H_res_bef should be captured"
    assert 'H_res' in layer.info, "H_res should be captured"
    print(f"  Captured H_res_bef shape: {layer.info['H_res_bef'].shape}")
    print(f"  Captured H_res shape: {layer.info['H_res'].shape}")
    
    layer.clear_info()
    assert len(layer.info) == 0, "clear_info should empty the dict"
    print("  Logging and clear_info: PASS")
    
    return True


def test_parameter_utilities():
    """Test the parameter printing utilities."""
    from hyper_conn import print_trainable_parameters, count_parameters, KromHC
    
    print("\nTesting parameter utilities...")
    
    layer = KromHC(4, dim=512)
    
    # Test count_parameters
    total = count_parameters(layer, trainable_only=True)
    assert total > 0, "Should have trainable parameters"
    print(f"  KromHC(4, dim=512) has {total:,} trainable parameters")
    
    # Test print_trainable_parameters (capture output)
    params = print_trainable_parameters(layer, show_shapes=True)
    assert len(params) > 0, "Should return parameter dict"
    print("  print_trainable_parameters: PASS")
    
    return True


def test_channel_first():
    """Test channel_first layout option."""
    from hyper_conn import KromHC
    
    print("\nTesting channel_first layout...")
    
    branch = nn.Linear(512, 512)
    
    # Standard layout (batch, seq, dim)
    layer_standard = KromHC(4, dim=512, channel_first=False, branch=branch)
    x_standard = torch.randn(8, 10, 512)
    y_standard = layer_standard(x_standard)
    assert y_standard.shape == (8, 10, 512)
    print("  channel_first=False: PASS")
    
    # Channel-first layout (batch, dim, seq) - use Conv1d as branch
    # Conv1d expects (batch, channels, length), applies convolution along length
    branch_cf = nn.Conv1d(512, 512, kernel_size=1)
    layer_cf = KromHC(4, dim=512, channel_first=True, branch=branch_cf)
    x_cf = torch.randn(8, 512, 10)
    y_cf = layer_cf(x_cf)
    assert y_cf.shape == (8, 512, 10)
    print("  channel_first=True: PASS")
    
    return True


def test_dropout():
    """Test that dropout is applied correctly."""
    from hyper_conn import KromHC
    
    print("\nTesting dropout...")
    
    branch = nn.Linear(512, 512)
    layer = KromHC(4, dim=512, dropout=0.5, branch=branch)
    layer.train()  # Enable dropout
    
    x = torch.randn(8, 10, 512)
    
    # Run multiple times - outputs should differ due to dropout
    y1 = layer(x)
    y2 = layer(x)
    
    # With 50% dropout, outputs should differ
    assert not torch.allclose(y1, y2), "Dropout should cause different outputs"
    print("  Dropout in training mode: PASS")
    
    layer.eval()  # Disable dropout
    y3 = layer(x)
    y4 = layer(x)
    assert torch.allclose(y3, y4), "Eval mode should be deterministic"
    print("  Deterministic in eval mode: PASS")
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("HYPER-CONNECTION MODULE TESTS")
    print("=" * 70)
    
    tests = [
        ("API Compatibility", test_api_compatibility),
        ("Inheritance Hierarchy", test_inheritance_hierarchy),
        ("Forward/Backward", test_forward_backward),
        ("ORTBP Optimizer Groups", test_ortbp_optimizer_groups),
        ("ORTBP Optimizer No Overrides", test_ortbp_optimizer_no_overrides_two_groups),
        ("Non-ORTBP Optimizer Groups", test_non_ortbp_optimizer_no_ortbp_groups),
        ("ORTBP Log Stats Smoke", test_ortbp_log_stats_smoke),
        ("Functional API", test_functional_api),
        ("Decorate Branch", test_decorate_branch),
        ("Double Stochasticity", test_double_stochasticity),
        ("MHCAnalysis Logging", test_mhc_analysis_logging),
        ("Parameter Utilities", test_parameter_utilities),
        ("Channel First", test_channel_first),
        ("Dropout", test_dropout),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results[name] = "PASS" if passed else "FAIL"
        except Exception as e:
            results[name] = f"ERROR: {e}"
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, result in results.items():
        status = "[PASS]" if result == "PASS" else "[FAIL]"
        print(f"  {status} {name}: {result}")
        if result != "PASS":
            all_passed = False
    
    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
