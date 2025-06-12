import torch

_num_sms = None

def set_num_sms(num_sms: int) -> None:
    """
    Set the maximum SM count for all GEMM kernels to use.

    Arguments:
        num_sms: the desired maximum SM count for all GEMM kernels to use.
    """
    global _num_sms
    assert 0 < num_sms <= torch.cuda.get_device_properties(device='cuda').multi_processor_count
    _num_sms = num_sms


def get_num_sms() -> int:
    """
    Get the current maximum limit of SM count for all GEMM kernels to use.
    If the count is never specified, the function will return the number of device SMs.

    Returns:
        Current maximum limit of SM count for all GEMM kernels to use.
    """
    global _num_sms
    if _num_sms is None:
        _num_sms = torch.cuda.get_device_properties(device='cuda').multi_processor_count
    return _num_sms
