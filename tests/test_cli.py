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
