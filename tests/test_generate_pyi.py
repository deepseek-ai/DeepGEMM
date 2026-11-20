"""Generated public stubs must preserve keyword-only argument defaults."""

import ast
from pathlib import Path
import unittest

from scripts.generate_pyi import (
    build_cpp_function_index, extract_m_def_statements,
    parse_mdef_and_attach_cpp_signatures, extract_cpp_signature_details,
    generate_pyi_function,
)


class KeywordOnlyStubTest(unittest.TestCase):
    def test_masked_bf16_bindings(self):
        root = Path(__file__).resolve().parents[1] / 'csrc'
        index = build_cpp_function_index(root)
        names = {'m_grouped_bf16_gemm_nt_masked', 'm_grouped_bf16_swiglu_nt_masked'}
        found = set()
        for item in extract_m_def_statements(root):
            item = extract_cpp_signature_details(parse_mdef_and_attach_cpp_signatures(item, index))
            for statement in item['m_def_statements']:
                name = statement['parsed']['python_function_name']
                if name not in names:
                    continue
                function = ast.parse(generate_pyi_function(statement)).body[0]
                self.assertEqual([p.arg for p in function.args.kwonlyargs], ['sm120_m_tile'])
                self.assertEqual(ast.literal_eval(function.args.kw_defaults[0]), 0)
                self.assertEqual(len(function.args.args), 6 if name.endswith('gemm_nt_masked') else 5)
                found.add(name)
        self.assertEqual(found, names)

    def test_private_moe_layout_default(self):
        root = Path(__file__).resolve().parents[1] / 'csrc'
        index = build_cpp_function_index(root)
        for item in extract_m_def_statements(root):
            item = extract_cpp_signature_details(parse_mdef_and_attach_cpp_signatures(item, index))
            for statement in item['m_def_statements']:
                if statement['parsed']['python_function_name'] != '_sm120_moe_small':
                    continue
                function = ast.parse(generate_pyi_function(statement)).body[0]
                self.assertEqual(len(function.args.args), 9)
                self.assertEqual([p.arg for p in function.args.kwonlyargs],
                                 ['gate_up_swapped', 'expert_ids'])
                self.assertEqual([ast.literal_eval(v) for v in function.args.kw_defaults],
                                 [False, False])
                return
        self.fail('Private MoE binding missing')


if __name__ == '__main__':
    unittest.main()
