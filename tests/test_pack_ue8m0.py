import random
import torch

from deep_gemm.utils import pack_ue8m0_to_int


def generate_ue8m0(m: int, n: int) -> torch.Tensor:
    # Positive powers of two: the sign bit and all mantissa bits are zero
    return torch.pow(2.0, torch.randint(-8, 9, (m, n), device='cuda').float())


def test_cuda_graph_capture() -> None:
    print('Testing CUDA graph capture/replay:')
    x_0, x_1 = generate_ue8m0(128, 32), generate_ue8m0(128, 32)
    eager_y_0, eager_y_1 = pack_ue8m0_to_int(x_0), pack_ue8m0_to_int(x_1)
    torch.cuda.synchronize()

    # NOTES: the eager calls above also serve as the pre-capture warm-up
    static_x = x_0.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        captured_y = pack_ue8m0_to_int(static_x)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured_y, eager_y_0)
    print(' > Capture/replay matches eager (bit-exact)')

    # Replay must recompute from the mutated static input, not expose stale capture-time data
    static_x.copy_(x_1)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured_y, eager_y_1)
    print(' > Replay after static-input mutation matches eager (bit-exact)')
    print()


def test_eager_validation() -> None:
    print('Testing eager malformed-input rejection:')
    for name, bad_value in (('nonzero mantissa', 1.5), ('negative sign', -2.0)):
        x = generate_ue8m0(128, 32)
        x[0, 0] = bad_value
        raised = False
        try:
            pack_ue8m0_to_int(x)
        except AssertionError:
            raised = True
        assert raised, f'Malformed input ({name}) must still be rejected in eager execution'
        print(f' > AssertionError raised as expected ({name})')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    test_cuda_graph_capture()
    test_eager_validation()
