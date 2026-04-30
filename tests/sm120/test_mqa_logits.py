"""SM120a MQA logits correctness and performance test.

Runs the existing test_attention.py test functions on SM120a,
covering both dense (ragged) and paged FP8 MQA logits.
"""
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')

from test_attention import test_mqa_logits, test_paged_mqa_logits

if __name__ == '__main__':
    test_mqa_logits()
    test_paged_mqa_logits()
    print('All MQA logits tests completed.')
