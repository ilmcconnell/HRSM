"""``hrsmpipe`` command line: ``hrsmpipe <config.toml>`` runs the pipeline; subcommands run stages."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Any

import typer
from typer.core import TyperGroup

from hrsmpipe import pipeline
from hrsmpipe.config import RunConfig, load_config, resolved_toml


class _BarePathGroup(TyperGroup):
    """``hrsmpipe cfg.toml ...`` is ``hrsmpipe run cfg.toml ...``."""

    def parse_args(self, ctx, args):
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            args = ["run", *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    cls=_BarePathGroup,
    help="HRSM soil-moisture pipeline. `hrsmpipe <config.toml>` runs every stage.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigArg = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="run config (TOML)")]
SetOpt = Annotated[
    list[str],
    typer.Option(
        "--set", "-s", help="override one key, e.g. --set gee.window_hours=6 (repeatable)"
    ),
]
SitesOpt = Annotated[
    str, typer.Option("--sites", help="comma-separated site_ids (sets sites.include)")
]
NameOpt = Annotated[str, typer.Option("--name", help="run.name")]
StartOpt = Annotated[str, typer.Option("--start", help="run.start (YYYY-MM-DD, UTC)")]
EndOpt = Annotated[str, typer.Option("--end", help="run.end (YYYY-MM-DD, UTC, exclusive)")]


def _parse_value(text: str) -> Any:
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text  # bare word -> string


def _build(
    config: Path | None,
    sets: list[str],
    sites: str = "",
    name: str = "",
    start: str = "",
    end: str = "",
) -> RunConfig:
    overrides: dict[str, Any] = {}
    for item in sets:
        if "=" not in item:
            raise typer.BadParameter(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = _parse_value(value.strip())
    if sites:
        overrides["sites.include"] = [int(s) for s in sites.split(",") if s.strip()]
    if name:
        overrides["run.name"] = name
    if start:
        overrides["run.start"] = start
    if end:
        overrides["run.end"] = end
    return load_config(config, overrides)


def _stage_command(stage: str, help_text: str):
    def command(
        config: ConfigArg,
        set_: SetOpt = [],  # noqa: B006 (typer reads the default; never mutated)
        sites: SitesOpt = "",
        name: NameOpt = "",
        start: StartOpt = "",
        end: EndOpt = "",
    ) -> None:
        cfg = _build(config, set_, sites, name, start, end)
        pipeline.run_stages(cfg, from_=stage, to=stage, force=stage)

    command.__name__ = stage.replace("-", "_")
    app.command(stage, help=help_text)(command)


_stage_command("fetch-static", "Sample static GEE layers (landcover, POLARIS, terrain) per site.")
_stage_command("fetch-rs", "Fetch ERA5-Land and Sentinel-1/2, HLSL30 hourly series; write rs/*.nc.")
_stage_command(
    "fetch-obs", "Load in-situ observations (Wisconet API/CSV archive/table); write obs.parquet."
)
_stage_command("features", "Build the per-sensor ML feature tables (features/*.parquet).")
_stage_command("infer", "Run the LightGBM + MAPIE-CQR models; write inference/<layer>.parquet.")
_stage_command(
    "postproc", "CDF-match sensors, derive sd, and write the postproc/*.nc fields and miu.csv."
)
_stage_command("forcing", "Build precipitation and PET forcing (mesonet + ERA5) and soil classes.")


@app.command("build-lgar")
def build_lgar(
    config: ConfigArg,
    set_: SetOpt = [],  # noqa: B006
    force: Annotated[
        bool, typer.Option("--force/--keep", help="rebuild even if the binaries exist")
    ] = True,
) -> None:
    """Compile the bundled LGAR and LGAR+EnKF C++ sources into <data_root>/bin/<platform>."""
    from hrsmpipe.lgar import build

    cfg = _build(config, set_)
    for kind, path in build.build_binaries(cfg, force=force).items():
        typer.echo(f"{kind}: {path}")


_stage_command(
    "lgar", "Run LGAR and LGAR+EnKF per site (builds the binaries if missing); merge outputs."
)
_stage_command(
    "evaluate", "Metrics vs observations, plots, and the optional comparison with the study."
)


@app.command("run")
def run(
    config_pos: Annotated[Path | None, typer.Argument(help="run config (TOML)")] = None,
    config_opt: Annotated[
        Path | None, typer.Option("--config", "-c", help="run config (TOML)")
    ] = None,
    set_: SetOpt = [],  # noqa: B006
    sites: SitesOpt = "",
    name: NameOpt = "",
    start: StartOpt = "",
    end: EndOpt = "",
    from_: Annotated[str, typer.Option("--from", help="first stage")] = "",
    to: Annotated[str, typer.Option("--to", help="last stage")] = "",
    force: Annotated[
        str, typer.Option("--force", help="rerun this stage and every later one")
    ] = "",
) -> None:
    """Run the pipeline (all stages by default); unchanged stages are skipped."""
    config = config_opt or config_pos
    if config is None:
        raise typer.BadParameter("give a config path (positional or --config)")
    if not config.exists():
        raise typer.BadParameter(f"config not found: {config}")
    cfg = _build(config, set_, sites, name, start, end)
    report = pipeline.run_stages(cfg, from_=from_ or None, to=to or None, force=force or None)
    typer.echo(f"ran: {', '.join(report.ran) or '-'}; skipped: {', '.join(report.skipped) or '-'}")


@app.command("init")
def init(path: Annotated[Path, typer.Argument(help="where to write the default config")]) -> None:
    """Write a config containing every key at its default."""
    if path.exists():
        raise typer.BadParameter(f"{path} already exists")
    path.write_text(resolved_toml(RunConfig()))
    typer.echo(f"wrote {path}")


@app.command("config-schema")
def config_schema() -> None:
    """Print every config key with its default."""
    typer.echo(resolved_toml(RunConfig()), nl=False)
