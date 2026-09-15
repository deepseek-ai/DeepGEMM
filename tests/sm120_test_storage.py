import torch


def native_matrix(values, padded=False, offset=0, extra=5, tma=True):
    rows, cols = values.shape
    quantum = 16 // values.element_size()
    stride = cols + extra if padded else cols
    if tma:
        stride = (stride + quantum - 1) // quantum * quantum
        offset *= quantum
    storage = torch.full((offset + rows * stride + quantum,), 19, dtype=values.dtype, device='cuda')
    matrix = storage.as_strided((rows, cols), (stride, 1), offset)
    matrix.copy_(values)
    if tma:
        assert matrix.data_ptr() % 16 == 0 and matrix.stride(0) * matrix.element_size() % 16 == 0
    return matrix, storage
