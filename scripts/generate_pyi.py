"""Generate deep_gemm/_C.pyi from TORCH_LIBRARY schemas and deep_gemm/_C.py wrappers."""
import ast
import re
from pathlib import Path

class BracketTracker:
    """Track () [] {} <> nesting for top-level comma/default splitting."""

    def __init__(self):
        self.paren = 0      # ()
        self.bracket = 0    # []
        self.brace = 0      # {}
        self.angle = 0      # <>

    def update(self, char: str):
        if char == '(':
            self.paren += 1
        elif char == ')':
            self.paren -= 1
        elif char == '[':
            self.bracket += 1
        elif char == ']':
            self.bracket -= 1
        elif char == '{':
            self.brace += 1
        elif char == '}':
            self.brace -= 1
        # Angle brackets < > are only treated as template delimiters
        # when not inside (), [], or {}
        elif char == '<' and self._in_top_level_of_other_brackets():
            self.angle += 1
        elif char == '>' and self.angle > 0 and self._in_top_level_of_other_brackets():
            self.angle -= 1

    def _in_top_level_of_other_brackets(self):
        return self.paren == 0 and self.bracket == 0 and self.brace == 0

    def is_top_level(self):
        return self.paren == 0 and self.bracket == 0 and self.brace == 0 and self.angle == 0


def split_top_level_commas(value: str) -> list[str]:
    """Split on commas not nested inside brackets."""
    parts = []
    current = []
    tracker = BracketTracker()
    for ch in value:
        if ch in '()[]{}<>':
            tracker.update(ch)
        if ch == ',' and tracker.is_top_level():
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())
    return parts


def find_top_level_equals(value: str) -> int:
    """Return index of top-level '=', or -1."""
    tracker = BracketTracker()
    for i, ch in enumerate(value):
        if ch in '()[]{}<>':
            tracker.update(ch)
        elif ch == '=' and tracker.is_top_level():
            return i
    return -1


def schema_type_to_python(type_str: str) -> str:
    """Map a TORCH schema type to a Python annotation."""
    type_str = type_str.strip()
    optional = type_str.endswith('?')
    if optional:
        type_str = type_str[:-1].strip()

    if type_str.startswith('Tensor'):
        py_type = 'torch.Tensor'
    elif type_str == 'int':
        py_type = 'int'
    elif type_str == 'bool':
        py_type = 'bool'
    elif type_str == 'float':
        py_type = 'float'
    elif type_str == 'str':
        py_type = 'str'
    elif type_str == 'int[]':
        py_type = 'list[int]'
    elif re.match(r'^int\[\d+\]$', type_str):
        # Fixed-size int[N] maps directly to a same-arity tuple.
        n = int(re.match(r'^int\[(\d+)\]$', type_str).group(1))
        py_type = f"tuple[{', '.join(['int'] * n)}]"
    elif type_str == 'ScalarType':
        py_type = 'torch.dtype'
    else:
        print(f'Warning: unrecognized schema type {type_str!r}, using Any')
        py_type = 'Any'

    if optional:
        return f'Optional[{py_type}]'
    return py_type


def schema_return_to_python(return_str: str) -> str:
    """Map a TORCH schema return type to a Python annotation."""
    return_str = return_str.strip()
    if return_str == '()':
        return 'None'
    if return_str in {'int', 'bool', 'float', 'str', 'Tensor'}:
        return {
            'int': 'int',
            'bool': 'bool',
            'float': 'float',
            'str': 'str',
            'Tensor': 'torch.Tensor',
        }[return_str]
    if return_str.startswith('(') and return_str.endswith(')'):
        inner = return_str[1:-1].strip()
        if not inner:
            return 'tuple[()]'
        parts = split_top_level_commas(inner)
        py_parts = [schema_return_to_python(part) for part in parts]
        return f'tuple[{", ".join(py_parts)}]'
    print(f'Warning: unrecognized schema return type {return_str!r}, using Any')
    return 'Any'


_SCALAR_TYPE_DEFAULTS = {
    'float': 'torch.float32',
    'float32': 'torch.float32',
    'double': 'torch.float64',
    'float64': 'torch.float64',
    'half': 'torch.float16',
    'float16': 'torch.float16',
    'bfloat16': 'torch.bfloat16',
    'byte': 'torch.uint8',
    'char': 'torch.int8',
    'short': 'torch.int16',
    'int': 'torch.int32',
    'long': 'torch.int64',
}


def schema_default_to_python(default_str: str) -> str:
    """Convert a TORCH schema default literal to a Python expression string."""
    default_str = default_str.strip()
    if default_str in {'None', 'True', 'False'}:
        return default_str
    if (default_str.startswith("'") and default_str.endswith("'")) or (
            default_str.startswith('"') and default_str.endswith('"')):
        return default_str
    if default_str in _SCALAR_TYPE_DEFAULTS:
        return _SCALAR_TYPE_DEFAULTS[default_str]
    if re.match(r'^[+-]?\d+$', default_str):
        return default_str
    if re.match(r'^[+-]?\d*\.\d+([eE][+-]?\d+)?$', default_str):
        return default_str
    print(f'Warning: unrecognized schema default {default_str!r}, using None')
    return 'None'


def parse_schema_arg(arg_str: str) -> dict:
    """Parse one TORCH schema argument such as 'Tensor? c=None'."""
    arg_str = arg_str.strip()
    if not arg_str:
        raise ValueError('empty schema argument')

    default = None
    eq_pos = find_top_level_equals(arg_str)
    if eq_pos != -1:
        default = schema_default_to_python(arg_str[eq_pos + 1:].strip())
        arg_str = arg_str[:eq_pos].strip()

    match = re.match(r'^(.+?)\s+([a-zA-Z_][a-zA-Z0-9_]*)$', arg_str)
    if not match:
        raise ValueError(f'could not parse schema argument: {arg_str!r}')
    return {
        'name': match.group(2),
        'py_type': schema_type_to_python(match.group(1)),
        'default': default,
    }


def parse_torch_schema(schema: str) -> dict:
    """Parse a TORCH schema into name, parameters, and return type."""
    arrow = schema.rfind(' -> ')
    if arrow == -1:
        raise ValueError(f'schema missing return type: {schema!r}')

    signature = schema[:arrow].strip()
    return_type = schema_return_to_python(schema[arrow + 4:].strip())

    open_paren = signature.find('(')
    if open_paren == -1:
        raise ValueError(f'schema missing argument list: {schema!r}')

    name = signature[:open_paren].strip()
    paren_depth = 0
    close_paren = -1
    for i in range(open_paren, len(signature)):
        if signature[i] == '(':
            paren_depth += 1
        elif signature[i] == ')':
            paren_depth -= 1
            if paren_depth == 0:
                close_paren = i
                break
    if close_paren == -1:
        raise ValueError(f'unclosed argument list in schema: {schema!r}')

    args_blob = signature[open_paren + 1:close_paren].strip()
    parameters = []
    if args_blob:
        for arg in split_top_level_commas(args_blob):
            parameters.append(parse_schema_arg(arg))

    return {
        'python_function_name': name,
        'parameters': parameters,
        'return_type': return_type,
    }


def _merge_named_pairs(parameters: list[dict], pairs: tuple[tuple[str, str], ...]) -> list[dict]:
    """Replace (tensor, scale_factor) arg pairs with one tuple-typed parameter."""
    by_name = {param['name']: param for param in parameters}
    sf_of = dict(pairs)
    drop = set(sf_of.values())

    out = []
    for param in parameters:
        if param['name'] in drop:
            continue
        sf_name = sf_of.get(param['name'])
        if sf_name is None:
            out.append(dict(param))
            continue
        base_optional = param['py_type'] == 'Optional[torch.Tensor]'
        sf_optional = by_name[sf_name]['py_type'] == 'Optional[torch.Tensor]'
        if base_optional and sf_optional:
            py_type = 'Optional[tuple[torch.Tensor, torch.Tensor]]'
        elif sf_optional:
            py_type = 'tuple[torch.Tensor, Optional[torch.Tensor]]'
        else:
            py_type = 'tuple[torch.Tensor, torch.Tensor]'
        out.append({
            'name': param['name'],
            'py_type': py_type,
            'default': None,
        })
    return out


def _is_tensor_schema_param(param: dict) -> bool:
    py_type = param['py_type']
    return py_type in {'torch.Tensor', 'Optional[torch.Tensor]'}


def _is_tensor_scale_factor_pair(base_name: str, sf_name: str) -> bool:
    if base_name == 'a' and sf_name == 'sfa':
        return True
    if base_name == 'b' and sf_name == 'sfb':
        return True
    return sf_name == f'{base_name}_sf'


def detect_tensor_sf_pairs(parameters: list[dict]) -> list[tuple[str, str]]:
    """Detect consecutive (tensor, scale_factor) arg pairs."""
    pairs = []
    i = 0
    while i < len(parameters) - 1:
        left, right = parameters[i], parameters[i + 1]
        if (
            _is_tensor_schema_param(left)
            and _is_tensor_schema_param(right)
            and _is_tensor_scale_factor_pair(left['name'], right['name'])
        ):
            pairs.append((left['name'], right['name']))
            i += 2
        else:
            i += 1
    return pairs


def _maybe_widen_int_list_value_param(parameters: list[dict]) -> None:
    """Single int[] value param in a Python wrapper usually accepts int | list[int]."""
    if len(parameters) == 1 and parameters[0]['name'] == 'value':
        if parameters[0]['py_type'] == 'list[int]':
            parameters[0]['py_type'] = 'int | list[int]'


def _promote_transform_sf_recipe_type(op_name: str, parameters: list[dict]) -> None:
    """transform_sf_into_required_layout's recipe is a std::variant, which int[N] can't express, so promote it here."""
    if op_name != 'transform_sf_into_required_layout':
        return
    recipe = next((p for p in parameters if p['name'] == 'recipe'), None)
    if recipe is not None and recipe['py_type'] == 'list[int]':
        recipe['py_type'] = 'tuple[int, int] | tuple[int, int, int]'


def adjust_for_c_py_wrapper(
    name: str,
    parameters: list[dict],
    wrapper_defaults: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    """Adjust flat schema params to match deep_gemm._C Python wrappers."""
    pairs = detect_tensor_sf_pairs(parameters)
    if pairs:
        parameters = _merge_named_pairs(parameters, tuple(pairs))

    _maybe_widen_int_list_value_param(parameters)
    _promote_transform_sf_recipe_type(name, parameters)

    return parameters


def format_ast_default(node: ast.AST) -> str:
    """Convert an AST default value node to a Python expression string."""
    if isinstance(node, ast.Constant):
        if node.value is None:
            return 'None'
        if isinstance(node.value, bool):
            return 'True' if node.value else 'False'
        if isinstance(node.value, str):
            return f'"{node.value}"'
        if isinstance(node.value, (int, float)):
            return repr(node.value)
    if isinstance(node, ast.Tuple):
        elts = ', '.join(format_ast_default(element) for element in node.elts)
        return f'({elts})'
    if isinstance(node, ast.List):
        elts = ', '.join(format_ast_default(element) for element in node.elts)
        return f'[{elts}]'
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f'-{format_ast_default(node.operand)}'
    return ast.unparse(node)


def extract_function_defaults(func_def: ast.FunctionDef) -> dict[str, str]:
    """Extract {param_name: default_expr} from a Python function definition."""
    defaults: dict[str, str] = {}
    args = func_def.args
    pos_args = args.args
    if args.defaults:
        first_default_idx = len(pos_args) - len(args.defaults)
        for idx, default_node in enumerate(args.defaults):
            defaults[pos_args[first_default_idx + idx].arg] = format_ast_default(default_node)
    for arg, default_node in zip(args.kwonlyargs, args.kw_defaults):
        if default_node is not None:
            defaults[arg.arg] = format_ast_default(default_node)
    return defaults


def _is_globals_update_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute) or node.func.attr != 'update':
        return False
    base = node.func.value
    if isinstance(base, ast.Name):
        return base.id == 'globals'
    if isinstance(base, ast.Call) and isinstance(base.func, ast.Name):
        return base.func.id == 'globals'
    return False


def parse_c_py_metadata(c_py_path: Path) -> dict[str, dict[str, str]]:
    """Parse wrapper defaults from deep_gemm/_C.py without importing it."""
    source = c_py_path.read_text(encoding='utf-8')
    module = ast.parse(source, filename=str(c_py_path))

    func_defaults: dict[str, dict[str, str]] = {}

    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef):
            func_defaults[node.name] = extract_function_defaults(node)

    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        if not _is_globals_update_call(node):
            continue
        if not node.args or not isinstance(node.args[0], ast.Dict):
            continue
        alias_dict = node.args[0]
        for key_node, value_node in zip(alias_dict.keys, alias_dict.values):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                continue
            alias_name = key_node.value
            if isinstance(value_node, ast.Name):
                if value_node.id in func_defaults:
                    func_defaults[alias_name] = func_defaults[value_node.id]

    return func_defaults


def extract_m_def_statements(root_path) -> list[str]:
    """Scan C++ sources under root_path for m.def(...) registrations."""
    statements = []
    extensions = {'.hpp', '.cpp', '.h', '.cc'}

    for file_path in sorted(Path(root_path).rglob('*')):
        if file_path.suffix.lower() not in extensions:
            continue
        if not file_path.is_file():
            continue

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f'Failed to read file {file_path}: {e}')
            continue

        m_def_list = []
        lines = content.splitlines(keepends=True)
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'm.def(' in line:
                stripped = line.lstrip()
                if stripped.startswith('//') or stripped.startswith('/*'):
                    i += 1
                    continue

                paren_count = 0
                j = i
                found_start = False
                while j < len(lines):
                    current_line = lines[j]
                    for k, char in enumerate(current_line):
                        if char == '(':
                            if not found_start and re.search(r'm\.def\s*\(', current_line[:k+1]):
                                found_start = True
                            if found_start:
                                paren_count += 1
                        elif char == ')':
                            if found_start:
                                paren_count -= 1
                                if paren_count == 0:
                                    full_stmt = ''.join(lines[i:j+1]).rstrip()
                                    m_def_list.append(full_stmt)
                                    i = j
                                    break
                    if paren_count <= 0 and found_start:
                        break
                    j += 1
            i += 1

        if m_def_list:
            statements.extend(m_def_list)

    return statements


def parse_m_def_statement(m_def_str):
    """Parse a TORCH_LIBRARY m.def(...) statement."""
    start = m_def_str.find('m.def(')
    if start == -1:
        raise ValueError(f'[{m_def_str}] Could not find m.def start position')

    paren_count = 0
    content_start = start + len('m.def(')
    content_end = -1
    for i in range(content_start, len(m_def_str)):
        ch = m_def_str[i]
        if ch == '(':
            paren_count += 1
        elif ch == ')':
            if paren_count == 0:
                content_end = i
                break
            else:
                paren_count -= 1
    if content_end == -1:
        raise ValueError(f'[{m_def_str}] m.def parentheses not closed')

    args_content = m_def_str[content_start:content_end]
    args_list = split_top_level_commas(args_content)
    if not args_list:
        raise ValueError(f'[{m_def_str}] m.def has no arguments')

    first = args_list[0].strip()
    str_match = re.match(r'^"([^"\\]*(?:\\.[^"\\]*)*)"', first)
    if not str_match:
        raise ValueError(f'[{m_def_str}] m.def first argument should be a string literal')

    return parse_torch_schema(str_match.group(1))


def apply_wrapper_defaults(name: str, parameters: list[dict], wrapper_defaults: dict[str, dict[str, str]]) -> list[dict]:
    """Overlay public API defaults from deep_gemm/_C.py onto schema-derived parameters."""
    by_name = wrapper_defaults.get(name, {})
    if not by_name:
        return parameters

    out = []
    for param in parameters:
        param = dict(param)
        if param['name'] in by_name:
            param['default'] = by_name[param['name']]
        out.append(param)
    return out


def generate_pyi_function(parsed, wrapper_defaults=None):
    """Generate a typed .pyi stub for one registered op."""
    py_name = parsed['python_function_name']
    parameters = adjust_for_c_py_wrapper(
        py_name,
        parsed['parameters'],
        wrapper_defaults=wrapper_defaults,
    )
    if wrapper_defaults:
        parameters = apply_wrapper_defaults(py_name, parameters, wrapper_defaults)
    return_type = parsed['return_type']

    param_lines = []
    for param in parameters:
        name = param['name']
        if param['default'] is not None:
            param_lines.append(f'    {name}: {param["py_type"]} = {param["default"]}')
        else:
            param_lines.append(f'    {name}: {param["py_type"]}')

    if param_lines:
        params_block = ',\n'.join(param_lines)
        return f'def {py_name}(\n{params_block}\n) -> {return_type}: ...'
    return f'def {py_name}() -> {return_type}: ...'


def generate_pyi_file_content(
    parsed_ops,
    module_name: str = 'my_module',
    wrapper_defaults=None,
):
    """Assemble the full .pyi file from parsed TORCH ops."""
    decls = []

    for parsed in parsed_ops:
        name = parsed['python_function_name']
        try:
            decl = generate_pyi_function(parsed, wrapper_defaults=wrapper_defaults)
        except Exception as e:
            decl = f'# ERROR: failed to generate stub for {name}: {e}'
        decls.append(decl)

    lines = [
        f'# Stubs for module: {module_name}',
        '',
        'from typing import Any, Optional',
        'import torch',
        '',
        '',
    ]

    for decl in decls:
        lines.extend([decl, '', ''])

    return '\n'.join(lines)


def generate_pyi_file(name, root, output_dir='.', c_py_path=None):
    """Generate stubs/<name>.pyi from csrc/ schemas and optional _C.py wrapper defaults."""
    m_def_statements = extract_m_def_statements(root)
    parsed_ops = [parse_m_def_statement(stmt) for stmt in m_def_statements]

    wrapper_defaults = {}
    if c_py_path is not None:
        c_py_path = Path(c_py_path)
        if c_py_path.is_file():
            wrapper_defaults = parse_c_py_metadata(c_py_path)
        else:
            print(f'Warning: wrapper file not found: {c_py_path}')

    pyi_content = generate_pyi_file_content(
        parsed_ops,
        module_name=name,
        wrapper_defaults=wrapper_defaults,
    )

    output_path = Path(output_dir) / f'{name}.pyi'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pyi_content)

    print(f'.pyi file generated: {output_path}')
