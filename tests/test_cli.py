import pytest

from neighbourhood_pulse import cli


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    assert "ingest" in capsys.readouterr().out


def test_no_command_is_an_error():
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2


def test_ingest_dispatches(monkeypatch):
    import neighbourhood_pulse.ingestion as ingestion_module

    called = {}

    class FakeIngestion:
        def run(self):
            called["ran"] = True

    monkeypatch.setattr(ingestion_module, "DataIngestion", FakeIngestion)
    cli.main(["ingest"])
    assert called == {"ran": True}


def test_transform_dispatches(monkeypatch):
    import neighbourhood_pulse.transformation as transformation_module

    called = {}

    class FakeTransformation:
        def run(self):
            called["ran"] = True

    monkeypatch.setattr(transformation_module, "DataTransformation", FakeTransformation)
    cli.main(["transform"])
    assert called == {"ran": True}


def test_ingest_failure_exits_1(monkeypatch):
    import neighbourhood_pulse.ingestion as ingestion_module

    class FailingIngestion:
        def run(self):
            raise ingestion_module.IngestionError("boom")

    monkeypatch.setattr(ingestion_module, "DataIngestion", FailingIngestion)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["ingest"])
    assert excinfo.value.code == 1


def test_train_dispatches_with_force(monkeypatch):
    import neighbourhood_pulse.pipeline as pipeline_module

    calls = {}

    def fake_run_train(force=False):
        calls["force"] = force
        return {"r2_linear": 0.4, "r2_xgboost": 0.44}

    monkeypatch.setattr(pipeline_module, "run_train", fake_run_train)
    cli.main(["train", "--force"])
    assert calls == {"force": True}


def test_briefs_without_api_key_exits_1(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["briefs"])
    assert excinfo.value.code == 1


def test_briefs_dispatches_with_key(monkeypatch):
    import neighbourhood_pulse.briefs as briefs_module

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    called = {}

    def fake_run_briefs(force):
        called.setdefault("force", force)

    monkeypatch.setattr(briefs_module, "run_briefs", fake_run_briefs)
    cli.main(["briefs", "--force"])
    assert called == {"force": True}
