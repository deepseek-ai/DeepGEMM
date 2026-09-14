import deep_gemm


def test_block_size_multiple_validation() -> None:
    invalid_values = (0, -1, (0, 1), (1, 0), (-1, 1), (1, -1))

    try:
        for value in invalid_values:
            try:
                deep_gemm.set_block_size_multiple_of(value)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f'Expected set_block_size_multiple_of({value}) to fail')
    finally:
        deep_gemm.set_block_size_multiple_of(1)


if __name__ == '__main__':
    test_block_size_multiple_validation()
