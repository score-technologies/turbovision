from click.testing import CliRunner

import scorevision


def test_root_help_lists_scorevision_commands_without_loading_lazy_groups(monkeypatch) -> None:
    """`sv --help` should work without importing lazy subcommand modules."""

    def fail_on_lazy_load(spec):
        raise AssertionError(f"unexpected lazy import for {spec.module_path}")

    monkeypatch.setattr(scorevision, "_load_click_command", fail_on_lazy_load)

    result = CliRunner().invoke(scorevision.app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "ScoreVision CLI" in result.output
    assert "-v, --verbosity" in result.output
    for command in [
        "runner",
        "push",
        "signer",
        "validate",
        "central-validator",
        "audit-validator",
        "manifest",
        "elements",
    ]:
        assert command in result.output


def test_manifest_help_loads_lazily_from_root() -> None:
    """`sv manifest --help` should load the manifest group only when requested."""

    result = CliRunner().invoke(scorevision.app, ["manifest", "--help"])

    assert result.exit_code == 0, result.output
    assert "Manage ScoreVision manifests." in result.output
    for command in ["create", "validate", "publish", "list", "current"]:
        assert command in result.output


def test_elements_help_loads_lazily_from_root() -> None:
    """`sv elements --help` should render without loading unrelated command modules."""

    result = CliRunner().invoke(scorevision.app, ["elements", "--help"])

    assert result.exit_code == 0, result.output
    assert "elements" in result.output
    assert "list" in result.output
