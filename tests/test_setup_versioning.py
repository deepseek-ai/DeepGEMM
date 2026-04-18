import importlib.util
from pathlib import Path
from unittest import mock


def load_setup_module():
    setup_path = Path(__file__).resolve().parents[1] / 'setup.py'
    spec = importlib.util.spec_from_file_location('deepgemm_setup', setup_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_exact_git_tag_version_prefers_matching_post_release():
    setup_module = load_setup_module()

    with mock.patch.object(
        setup_module.subprocess,
        'check_output',
        return_value=b'v2.1.1\nv2.1.1.post3\nv9.9.9\n',
    ):
        assert setup_module.get_exact_git_tag_version('2.1.1') == '2.1.1.post3'


def test_get_package_version_uses_exact_post_release_tag_when_local_suffix_disabled():
    setup_module = load_setup_module()

    with (
        mock.patch.object(setup_module, 'DG_USE_LOCAL_VERSION', False),
        mock.patch.object(setup_module, 'get_public_version', return_value='2.1.1'),
        mock.patch.object(
            setup_module,
            'get_exact_git_tag_version',
            return_value='2.1.1.post3',
        ),
    ):
        assert setup_module.get_package_version() == '2.1.1.post3'


def test_get_package_version_appends_local_revision_after_resolving_exact_tag():
    setup_module = load_setup_module()

    with (
        mock.patch.object(setup_module, 'DG_USE_LOCAL_VERSION', True),
        mock.patch.object(setup_module, 'get_public_version', return_value='2.1.1'),
        mock.patch.object(
            setup_module,
            'get_exact_git_tag_version',
            return_value='2.1.1.post3',
        ),
        mock.patch.object(
            setup_module.subprocess,
            'check_output',
            side_effect=[b'', b'abc123\n'],
        ),
    ):
        assert setup_module.get_package_version() == '2.1.1.post3+abc123'
