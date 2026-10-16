import pytest

from hrsmpipe import cli, manifest, pipeline
from hrsmpipe.config import load_config
from hrsmpipe.paths import RunPaths


def test_every_stage_is_registered_and_fingerprinted():
    registry = pipeline._registry()
    assert set(registry) == set(pipeline.STAGES)
    assert set(manifest.STAGE_SECTIONS) == set(pipeline.STAGES)
    assert (
        [s for s in pipeline.STAGES].index("forcing")
        < pipeline.STAGES.index("lgar")
        < pipeline.STAGES.index("evaluate")
    )


def test_every_stage_has_a_cli_subcommand():
    from typer.main import get_command

    commands = set(get_command(cli.app).commands)
    assert set(pipeline.STAGES) <= commands
    assert {"init", "config-schema", "build-lgar", "run"} <= commands


def test_stage_inputs_and_outputs_resolve_for_a_config(cfg, data_root):
    from hrsmpipe.sites import load_sites

    sites = load_sites(cfg)
    paths = RunPaths(data_root, cfg.run.name)
    for name, stage in pipeline._registry().items():
        assert all(hasattr(p, "exists") for p in stage.inputs(cfg, sites)), name
        assert stage.outputs(cfg, paths), name


def test_force_reruns_that_stage_and_everything_after():
    forced = manifest.stages_to_run(pipeline.STAGES, "forcing")
    assert forced == {"forcing", "lgar", "evaluate"}
    with pytest.raises(ValueError):
        manifest.stages_to_run(pipeline.STAGES, "nope")


def test_run_stages_skips_unchanged_and_forces_downstream(monkeypatch, cfg, data_root):
    """Fake registry: only the 'forcing' stage onwards runs after --force forcing."""
    calls = []
    ran_outputs = {}

    def fake_registry():
        reg = {}
        for name in pipeline.STAGES:
            out = data_root / "runs" / cfg.run.name / f"{name}.out"

            def make(name=name, out=out):
                def run(_cfg):
                    calls.append(name)
                    out.write_text("x")

                return run

            reg[name] = pipeline.Stage(name, make(), lambda c, rp, out=out: [out])
            ran_outputs[name] = out
        return reg

    monkeypatch.setattr(pipeline, "_registry", fake_registry)
    first = pipeline.run_stages(cfg)
    assert first.ran == list(pipeline.STAGES) and calls == list(pipeline.STAGES)
    calls.clear()
    again = pipeline.run_stages(cfg)
    assert again.ran == [] and again.skipped == list(pipeline.STAGES) and calls == []
    pipeline.run_stages(cfg, force="forcing")
    assert calls == ["forcing", "lgar", "evaluate"]
    calls.clear()
    ran_outputs[
        "features"
    ].unlink()  # a missing output also triggers a rerun (and everything after it)
    pipeline.run_stages(cfg)
    assert calls == list(pipeline.STAGES[pipeline.STAGES.index("features") :])
    # the resolved config is written before any work and records provenance
    text = (data_root / "runs" / cfg.run.name / "config.resolved.toml").read_text()
    assert "[provenance]" in text and "hrsmpipe_version" in text


def test_run_stages_site_restriction_is_recorded_in_resolved_config(monkeypatch, cfg, data_root):
    monkeypatch.setattr(
        pipeline,
        "_registry",
        lambda: {n: pipeline.Stage(n, lambda c: None, lambda c, r: []) for n in pipeline.STAGES},
    )
    pipeline.run_stages(cfg, from_="fetch-static", to="fetch-static", sites=[7, 47])
    text = (data_root / "runs" / cfg.run.name / "config.resolved.toml").read_text()
    assert "include = [\n    7,\n    47,\n]" in text or "include = [7, 47]" in text
    assert load_config(
        data_root / "runs" / cfg.run.name / "config.resolved.toml"
    ).sites.include == [7, 47]
