import pytest
from r2v2_deploy.cli.delivery import expand, load_assets


def test_unresolved_asset_paths_are_rejected(monkeypatch):
    monkeypatch.delenv("R2V2_UNCONFIGURED_ASSET", raising=False)
    with pytest.raises(ValueError):
        expand("${R2V2_UNCONFIGURED_ASSET}/model.onnx")


def test_nested_paths_expand_without_global_sys_path_changes(monkeypatch):
    monkeypatch.setenv("R2V2_TEST_ASSETS", "/assets")
    assert expand({"paths": ["${R2V2_TEST_ASSETS}/model.onnx"]}) == {
        "paths": ["/assets/model.onnx"]
    }


def test_unknown_manifest_refuses_to_load(tmp_path):
    path = tmp_path / "assets.json"
    path.write_text('{"schema":"unknown"}')
    with pytest.raises(ValueError):
        load_assets(path)
