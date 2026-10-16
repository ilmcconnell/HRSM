"""LGAR settings file (``key: value`` lines read by the C++ ``ReadSettingsFromFile``)."""

from __future__ import annotations

from pathlib import Path

from hrsmpipe.config import RunConfig

# Order of the notebook's settings dict. `soil_params_file` is only written when non-empty.
KEY_ORDER = [
    "verbosity",
    "site_ID",
    "num_hours",
    "layer_thickness_cm",
    "giuh_ordinates",
    "layer_soil_type",
    "timestep_h",
    "endtime_s",
    "field_capacity_psi",
    "wilting_point_psi",
    "initial_psi",
    "initial_sm_perturb",
    "soil_params_file",
    "TO_enabled",
    "adaptive_timestep",
    "free_drainage_enabled",
    "use_closed_form_G",
    "root_zone_depth_cm",
    "mbal_tol",
    "ponded_depth_max_cm",
    "forcing_resolution_h",
    "max_valid_soil_types",
]


def _fmt_scalar(v) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def format_settings(settings: dict) -> str:
    """Lists as ``[a, b, c]``, booleans lowercase; one ``key: value`` per line."""
    lines = []
    for key, value in settings.items():
        if isinstance(value, (list, tuple)):
            text = "[" + ", ".join(_fmt_scalar(x) for x in value) + "]"
        else:
            text = _fmt_scalar(value)
        lines.append(f"{key}: {text}")
    return "\n".join(lines) + "\n"


def write_settings(path: Path, settings: dict) -> None:
    Path(path).write_text(format_settings(settings))


def site_settings(cfg: RunConfig, site_id: int, num_hours: int, layer_soil_type: list[int]) -> dict:
    """All keys the C++ parser reads; ``site_ID``, ``num_hours``, ``endtime_s``, ``layer_soil_type``
    are derived at run time, everything else comes from ``[lgar]`` and ``[lgar.site_overrides.<id>]``."""
    values = cfg.lgar.model_dump()
    values.update(cfg.lgar.site_overrides.get(int(site_id), {}))
    values["site_ID"] = int(site_id)
    values["num_hours"] = int(num_hours)
    values["endtime_s"] = 3600 * int(num_hours)
    values.setdefault("layer_soil_type", None)
    values["layer_soil_type"] = (
        values["layer_soil_type"]
        if values["layer_soil_type"] is not None
        else [int(s) for s in layer_soil_type]
    )
    out = {}
    for key in KEY_ORDER:
        if key == "soil_params_file" and not values.get(key):
            continue
        out[key] = values[key]
    return out
