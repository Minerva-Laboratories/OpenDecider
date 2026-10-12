import pytest
import torch
import torch.nn as nn

from opendecider.quant import Int8Linear, fast_int8_kernels_


def test_fast_kernels_noop_on_cpu(monkeypatch):
    """CPU (and OPENDECIDER_INT8_GEMM=dequant) keep the dequantize path untouched."""
    class BB:
        device = torch.device("cpu")
        lm = nn.Module()
    BB.lm.layers = nn.Sequential(Int8Linear(nn.Linear(16, 8)))
    assert fast_int8_kernels_(BB) == 0 and isinstance(BB.lm.layers[0], Int8Linear)
    monkeypatch.setenv("OPENDECIDER_INT8_GEMM", "dequant")
    assert fast_int8_kernels_(BB) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GemLite needs CUDA")
def test_gemlite_holds_the_same_int8_weights():
    """The fused layer stores the int8 weights exactly (q+128, zero point 128): its output matches the dequantize path
    up to bf16 rounding inside the kernel."""
    pytest.importorskip("gemlite")
    from opendecider.quant import int8_to_gemlite
    torch.manual_seed(0)
    ref = Int8Linear(nn.Linear(512, 256, bias=False).cuda().to(torch.bfloat16))
    g = int8_to_gemlite(ref)
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    y0, y1 = ref(x).float(), g(x).float()
    assert ((y1 - y0).norm() / y0.norm()).item() < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GemLite needs CUDA")
def test_dense_path_reads_the_packed_weights_exactly():
    """Large batches expand GemLite's packed buffer: the weight must equal the int8 layer's exactly, and both paths of
    the dispatching layer must agree up to bf16 kernel rounding."""
    pytest.importorskip("gemlite")
    from opendecider.quant import Int8GemLite, int8_to_gemlite
    torch.manual_seed(0)
    ref = Int8Linear(nn.Linear(512, 256).cuda().to(torch.bfloat16))
    lay = Int8GemLite(int8_to_gemlite(ref), dense_from=100)
    assert torch.equal(lay.dense_weight().t(), ref.weight)
    x = torch.randn(150, 512, device="cuda", dtype=torch.bfloat16)
    y_dense = lay(x).float()
    lay.dense_from = None
    y_fused = lay(x).float()
    y_ref = ref(x).float()
    assert ((y_dense - y_ref).norm() / y_ref.norm()).item() < 1e-2
    assert ((y_fused - y_ref).norm() / y_ref.norm()).item() < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GemLite needs CUDA")
def test_fused_kernels_keep_module_objects_and_hooks():
    """VeRA adapters are forward hooks on the Int8Linear objects: attaching fused kernels must keep those objects (and
    so the hooks) and free the int8 copy, not replace the modules."""
    pytest.importorskip("gemlite")
    from opendecider.quant import int8_to_gemlite_
    torch.manual_seed(0)
    lin = Int8Linear(nn.Linear(256, 128, bias=False).cuda().to(torch.bfloat16))
    net = nn.Sequential(lin)
    lin.register_forward_hook(lambda mod, args, out: out + 100.0)
    x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    y0 = net(x).float()
    assert int8_to_gemlite_(net) == 1
    assert net[0] is lin and lin.qweight.numel() == 0
    y1 = net(x).float()
    assert (y1 - 100.0).abs().mean() < 50 and ((y1 - y0).norm() / y0.norm()).item() < 1e-2


def test_smoothquant_wrapper_is_exact_math():
    """SmoothedLinear(inner with W*s)(x) == the original linear(x); scales follow max|X|^a / max|W|^(1-a)."""
    import torch
    import torch.nn as nn
    from opendecider.quant import SmoothedLinear, smooth_scales
    torch.manual_seed(0)
    lin = nn.Linear(16, 8, bias=True).double()
    x = torch.randn(32, 16, dtype=torch.float64) * torch.logspace(-1, 2, 16, dtype=torch.float64)   # outlier channels
    s = smooth_scales(x.abs().amax(0), lin.weight.data, 0.5).double()
    assert torch.allclose(s, (x.abs().amax(0) ** 0.5 / lin.weight.abs().amax(0) ** 0.5).clamp(1e-3, 1e3), rtol=1e-4)
    inner = nn.Linear(16, 8, bias=True).double()
    inner.weight.data = lin.weight.data * s[None, :]
    inner.bias.data = lin.bias.data.clone()
    sm = SmoothedLinear(inner, s)
    sm.inv_s = (1.0 / s)                                                  # float64 for the exactness check
    assert torch.allclose(sm(x), lin(x), atol=1e-10)
    # smoothing flattens the activation range across channels (what makes per-token int8 easier)
    xs = x / s
    assert xs.abs().amax(0).max() / xs.abs().amax(0).min() < x.abs().amax(0).max() / x.abs().amax(0).min()
