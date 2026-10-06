import torch

from deep_gemm.utils import pack_ue8m0_to_int, unpack_ue8m0_from_int


def assert_raises(fn) -> None:
    try:
        fn()
    except AssertionError:
        return
    raise AssertionError('Expected an `AssertionError`')


def random_ue8m0(shape: tuple, device: str = 'cpu') -> torch.Tensor:
    # Zero-mantissa floats: `2 ** k` with random exponents in the normal range
    exp = torch.randint(1, 254, shape, dtype=torch.int32, device=device)
    return (exp << 23).view(torch.float)


def test_pack_values() -> None:
    print('Testing `pack_ue8m0_to_int` packing values:')
    x = torch.tensor([1.0, 2.0, 0.5, 128.0])
    packed = pack_ue8m0_to_int(x)
    # Four UE8M0 bytes packed little-endian into one int32
    assert packed.shape == (1, ) and packed.dtype == torch.int
    assert packed.view(torch.uint8).tolist() == [127, 128, 126, 134]
    # Unpacking restores the `2 ** k` values exactly
    assert torch.equal(unpack_ue8m0_from_int(packed), x)
    print(' > passed')


def test_pack_shapes() -> None:
    print('Testing `pack_ue8m0_to_int` across shapes:')
    for shape in ((4, ), (8, 4), (3, 2, 8)):
        x = random_ue8m0(shape)
        packed = pack_ue8m0_to_int(x)
        assert packed.shape == (*shape[:-1], shape[-1] // 4) and packed.dtype == torch.int
        assert torch.equal(unpack_ue8m0_from_int(packed), x)
    print(' > passed')


def test_eager_validation() -> None:
    print('Testing `pack_ue8m0_to_int` eager validation:')
    # A negative sign bit and a non-zero mantissa must both be rejected eagerly
    assert_raises(lambda: pack_ue8m0_to_int(torch.full((4, ), -2.0)))
    assert_raises(lambda: pack_ue8m0_to_int(torch.full((4, ), 1.5)))
    assert_raises(lambda: pack_ue8m0_to_int(torch.empty(3)))
    assert_raises(lambda: pack_ue8m0_to_int(torch.empty(4, dtype=torch.bfloat16)))
    print(' > passed')


def test_cuda_graph_capture() -> None:
    print('Testing `pack_ue8m0_to_int` under CUDA graph capture:')
    x = random_ue8m0((2, 3, 8))
    if torch.cuda.is_available():
        x_cuda = random_ue8m0((2, 3, 8), device='cuda')
        # Warm up on a side stream, then capture: the value checks used to force
        # a device-to-host sync here and failed the capture
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                pack_ue8m0_to_int(x_cuda)
        torch.cuda.current_stream().wait_stream(side_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = pack_ue8m0_to_int(x_cuda)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, pack_ue8m0_to_int(x_cuda))
        # Eager validation is preserved outside capture
        assert_raises(lambda: pack_ue8m0_to_int(torch.full((4, ), 1.5, device='cuda')))
        print(' > capture and replay match the eager packing on CUDA')
    else:
        print(' > skipped, CUDA is not available')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)

    test_pack_values()
    test_pack_shapes()
    test_eager_validation()
    test_cuda_graph_capture()
