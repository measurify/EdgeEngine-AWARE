#!/usr/bin/env python3
"""Turn a ``PolicyBundle`` JSON file into a C header for the firmware runtime.

    python tools/export_c.py examples/bundles/ppo_default.json -o firmware/eea_policy_data.h

The header defines

* ``EEA_PROFILE``       (``eea_profile_t``)     - the node profile of the bundle,
* ``EEA_RULE_PARAMS``   (``eea_rule_params_t``) - for a rule-based bundle (``EEA_HAS_RULE_PARAMS``),
* ``EEA_MLP``           (``eea_mlp_t``)         - for an MLP bundle, with the weight arrays
  (``EEA_HAS_MLP``),
* ``EEA_BUNDLE_*`` string macros with the bundle metadata (version, export time, notes),

so that ``firmware/eea_node.c`` reproduces exactly the observation, the
feasibility rule and the decision function that were validated in Python.
Constants are written with enough digits to round-trip exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MAX_MODES = 4

OBS_DEFAULTS = {
    "harvest_ref_power_w": 0.003,
    "age_scale_s": 86400.0,
    "harvest_ewma_alpha": 0.2,
    "link_ewma_alpha": 0.2,
    "importance_scale": 0.10,
    "path_loss_min_db": 110.0,
    "path_loss_max_db": 170.0,
}
PROFILE_DEFAULTS = {"timestep_s": 900.0, "baseline_power_w": 200e-6, "reserve_soc": 0.02, "ack_available": True, "critical_is_upper": False}
RULE_FIELDS = [
    "soc_critical", "soc_low", "deep_eco_interval_h", "soc_high", "harvest_strong", "report_interval_h",
    "report_interval_eco_factor", "eco_sensing_level", "report_interval_generous_factor", "check_interval_h",
    "event_delta", "importance_immediate", "importance_delta", "retry_age_h", "age_scale_h",
    "link_margin_target_db", "link_quality_escalate", "retry_min_link_quality",
]
ACTIVATIONS = {"linear": "EEA_ACT_LINEAR", "tanh": "EEA_ACT_TANH", "relu": "EEA_ACT_RELU"}


def c_double(x: float) -> str:
    r = repr(float(x))
    if r in ("inf", "-inf", "nan"):
        raise ValueError(f"non-finite constant {x}")
    return r if ("." in r or "e" in r) else r + ".0"


def c_float(x: float) -> str:
    return c_double(x) + "f"


def c_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def pad(values: list[float], n: int) -> list[float]:
    values = list(values)
    if len(values) > n:
        raise ValueError(f"at most {n} values supported, got {len(values)}")
    return values + [0.0] * (n - len(values))


def profile_block(profile: dict[str, Any]) -> str:
    obs = dict(OBS_DEFAULTS)
    obs.update(profile.get("observation") or {})
    prof = dict(PROFILE_DEFAULTS)
    prof.update({k: v for k, v in profile.items() if k != "observation"})
    n_modes = len(prof["tx_energy_j"])
    if n_modes > MAX_MODES:
        raise ValueError(f"EEA_MAX_MODES is {MAX_MODES}, the bundle has {n_modes} radio modes")
    lines = ["static const eea_profile_t EEA_PROFILE = {"]
    lines.append("    .sensing_energy_j = {" + ", ".join(c_double(v) for v in prof["sensing_energy_j"]) + "},")
    lines.append("    .sensing_noise_std = {" + ", ".join(c_double(v) for v in prof["sensing_noise_std"]) + "},")
    lines.append(f"    .n_modes = {n_modes},")
    for key in ("tx_energy_j", "tx_power_dbm", "sensitivity_dbm"):
        lines.append(f"    .{key} = {{" + ", ".join(c_double(v) for v in pad(prof[key], MAX_MODES)) + "},")
    lines.append(f"    .reference_mode = {int(prof['reference_mode'])},")
    for key in ("warning_threshold", "critical_threshold", "timestep_s", "baseline_power_w", "reserve_soc"):
        lines.append(f"    .{key} = {c_double(prof[key])},")
    lines.append(f"    .ack_available = {'true' if prof['ack_available'] else 'false'},")
    lines.append(f"    .critical_is_upper = {'true' if prof.get('critical_is_upper', False) else 'false'},")
    for key in OBS_DEFAULTS:
        lines.append(f"    .{key} = {c_double(obs[key])},")
    lines.append("};")
    return "\n".join(lines)


def rule_block(model: dict[str, Any]) -> str:
    lines = ["#define EEA_HAS_RULE_PARAMS 1", "static const eea_rule_params_t EEA_RULE_PARAMS = {"]
    for key in RULE_FIELDS:
        v = model[key]
        if key == "report_interval_h":
            lines.append("    .report_interval_h = {" + ", ".join(c_double(x) for x in v) + "},")
        elif key == "eco_sensing_level":
            lines.append(f"    .eco_sensing_level = {int(v)},")
        else:
            lines.append(f"    .{key} = {c_double(v)},")
    lines.append("};")
    return "\n".join(lines)


def mlp_block(model: dict[str, Any]) -> str:
    layers = model["layers"]
    if len(layers) > 4:
        raise ValueError("EEA_MAX_LAYERS is 4")
    out: list[str] = ["#define EEA_HAS_MLP 1"]
    for i, layer in enumerate(layers):
        W, b = layer["W"], layer["b"]
        n_out, n_in = len(W), len(W[0])
        if n_out > 128 or n_in > 128:
            raise ValueError("EEA_MAX_UNITS is 128")
        out.append(f"/* layer {i}: {n_in} -> {n_out}, {layer['activation']} */")
        out.append(f"static const float EEA_L{i}_W[{n_out * n_in}] = {{")
        for row in W:
            out.append("    " + ", ".join(c_float(v) for v in row) + ",")
        out.append("};")
        out.append(f"static const float EEA_L{i}_b[{n_out}] = {{" + ", ".join(c_float(v) for v in b) + "};")
    kind = "EEA_OUT_FLAT_Q_VALUES" if model["output"] == "flat_q_values" else "EEA_OUT_MULTIDISCRETE_LOGITS"
    split = model.get("output_split") or [3, len(layers[-1]["b"]) - 3]
    n_modes = int(model.get("n_modes", split[1] - 1))
    out.append("static const eea_mlp_t EEA_MLP = {")
    out.append(f"    .n_layers = {len(layers)},")
    out.append("    .layers = {")
    for i, layer in enumerate(layers):
        n_out, n_in = len(layer["W"]), len(layer["W"][0])
        out.append(f"        {{.in_dim = {n_in}, .out_dim = {n_out}, .W = EEA_L{i}_W, .b = EEA_L{i}_b, .activation = {ACTIVATIONS[layer['activation']]}}},")
    out.append("    },")
    out.append(f"    .output_kind = {kind},")
    out.append(f"    .output_split = {{{int(split[0])}, {int(split[1])}}},")
    out.append(f"    .n_modes = {n_modes},")
    out.append("};")
    return "\n".join(out)


def export_header(bundle: dict[str, Any], source: str = "") -> str:
    meta = bundle.get("metadata", {})
    model = bundle.get("model", {})
    parts = [
        "/* Generated by tools/export_c.py - do not edit.",
        f" * source : {source}",
        f" * policy : {bundle.get('policy_type')}",
        f" * version: {meta.get('edgeengine_aware_version', '?')}  exported {meta.get('exported_at', '?')}",
        f" * notes  : {meta.get('notes', '')}",
        " */",
        "#ifndef EEA_POLICY_DATA_H",
        "#define EEA_POLICY_DATA_H",
        '#include "eea_node.h"',
        f"#define EEA_BUNDLE_POLICY_TYPE {c_string(str(bundle.get('policy_type')))}",
        f"#define EEA_BUNDLE_VERSION {c_string(str(meta.get('edgeengine_aware_version', '')))}",
        f"#define EEA_BUNDLE_EXPORTED_AT {c_string(str(meta.get('exported_at', '')))}",
        f"#define EEA_BUNDLE_NOTES {c_string(str(meta.get('notes', '')))}",
        f"#define EEA_BUNDLE_OBS_DIM {len(bundle.get('observation_names', []))}",
        "",
        profile_block(bundle["profile"]),
        "",
    ]
    if bundle.get("policy_type") == "rule_based" or (model and "soc_low" in model):
        parts.append(rule_block(model))
    elif model.get("type") == "mlp" or "layers" in model:
        parts.append(mlp_block(model))
    else:
        parts.append("/* no decision function in this bundle (profile only) */")
    parts += ["", "#endif /* EEA_POLICY_DATA_H */", ""]
    return "\n".join(p for p in parts if p is not None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("firmware/eea_policy_data.h"))
    args = ap.parse_args(argv)
    bundle = json.loads(args.bundle.read_text())
    header = export_header(bundle, source=str(args.bundle))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header)
    n_obs = len(bundle.get("observation_names", []))
    if n_obs and n_obs != 18:
        print(f"warning: bundle has {n_obs} observation components, the C runtime expects 18", file=sys.stderr)
    print(f"wrote {args.out} ({len(header.splitlines())} lines)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
