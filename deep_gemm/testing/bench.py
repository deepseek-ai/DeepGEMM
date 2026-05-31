import os
import sys
import torch
import torch.distributed as dist
from typing import Callable, Optional


def bench(fn, num_warmups: int = 5, num_tests: int = 10,
          high_precision: bool = False):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
    cache.zero_()

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Add a large kernel to eliminate the CPU launch overhead
    if high_precision:
        x = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
        y = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
        x @ y

    # Testing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(num_tests):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / num_tests / 1e3


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(fn, kernel_names, num_tests: int = 30,
                 suppress_kineto_output: bool = False,
                 trace_path: str = None, flush_l2: bool = True,
                 with_multiple_kernels: bool = False,
                 barrier: Optional[Callable] = None):
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)

    # Skip profiling
    # Conflict with Nsight Systems, Nsight Compute and Compute Sanitizer
    if int(os.environ.get('DG_USE_NVIDIA_TOOLS', 0)):
        return (1, ) * len(kernel_names) if is_tuple else 1

    # Flush L2 between timed calls. Large MoE benchmark runs can use shared nodes
    # with limited free memory, so allow lowering only the flush buffer while
    # preserving the default behavior.
    flush_l2_bytes = int(os.environ.get('DG_BENCH_FLUSH_L2_BYTES', str(int(8e9))))
    flush_l2_size = max(0, flush_l2_bytes // 4)

    # Warm up once before profiling.
    fn()

    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule, acc_events=True)
        with profiler:
            for i in range(2):
                for _ in range(num_tests):
                    if flush_l2 and flush_l2_size > 0:
                        torch.empty(flush_l2_size, dtype=torch.int, device='cuda').zero_()
                    if barrier is not None:
                        # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                        # noinspection PyProtectedMember
                        torch.cuda._sleep(int(2e7))  # ~10ms
                        barrier()
                    fn()
                torch.cuda.synchronize()
                profiler.step()

    # Parse the profiling table
    report_split_kernels = int(os.environ.get('DG_SM90_MOE_REPORT_SPLIT_KERNELS', 0)) != 0
    max_name_column_width = int(os.environ.get('DG_BENCH_MAX_NAME_COLUMN_WIDTH', 100))
    if report_split_kernels and with_multiple_kernels:
        max_name_column_width = max(max_name_column_width, 512)
    prof_lines = profiler.key_averages().table(
        sort_by='cuda_time_total',
        max_name_column_width=max_name_column_width).split('\n')
    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
    if not with_multiple_kernels:
        for name in kernel_names:
            assert sum([name in line for line in prof_lines]) <= 1, f'Errors of the kernel {name} in the profiling table {prof_lines}'
    elif report_split_kernels:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        for line in prof_lines:
            if any(name in line for name in kernel_names):
                phase = '?'
                if 'sm90_fp8_mega_moe_impl<' in line:
                    template_args = line.split('sm90_fp8_mega_moe_impl<', 1)[1].split('>(void', 1)[0]
                    args = [arg.strip() for arg in template_args.split(',')]
                    if len(args) > 29:
                        phase_arg = args[29]
                        phase = {'1u': 'l1', '2u': 'l2', '0u': 'one'}.get(phase_arg, phase_arg)
                fields = line.split()
                cuda_time = fields[-2] if len(fields) >= 2 else 'unknown'
                count = fields[-1] if fields else 'unknown'
                print(f' > split_kernel rank={rank} phase={phase} cuda_time={cuda_time} count={count}')
                if int(os.environ.get('DG_SM90_MOE_REPORT_SPLIT_KERNELS_RAW', 0)) != 0:
                    print(f' > split_kernel_raw rank={rank}: {" ".join(fields)}')

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return average kernel times
    units = {'ms': 1e3, 'us': 1e6}
    kernel_times = []
    for name in kernel_names:
        total_time = 0
        total_num = 0
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                num_str = line.split()[-1]
                for unit, scale in units.items():
                    if unit in time_str:
                        total_time += float(time_str.replace(unit, '')) / scale * int(num_str)
                        total_num += int(num_str)
                        break
        if total_num > 0 and with_multiple_kernels:
            # Multiple matching kernels can belong to one logical benchmarked op
            # (e.g. split MegaMoE L1/L2). Report summed CUDA time per fn() call.
            kernel_times.append(total_time / num_tests)
        else:
            kernel_times.append(total_time / total_num if total_num > 0 else 0)

    return tuple(kernel_times) if is_tuple else kernel_times[0]
