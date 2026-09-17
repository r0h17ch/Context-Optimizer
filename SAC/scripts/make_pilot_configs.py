"""Generate the pilot run directories described in plan-v2 Part 3.

Every run starts SFT from the authors' released 15x pretrain adapter and differs from R0
only by the keys below, so the arms are paired at the training level: same seed (12345),
same data order (the shuffle happens once at data-prep time under random.seed(0)), same
initialisation, same truncation. Only the loss term changes.

    python scripts/make_pilot_configs.py [--steps 5000] [--force]
"""
import argparse
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SAC_HOME = os.path.dirname(HERE)
BASE = os.path.join(SAC_HOME, "experiment/release_15x")
GATE_REPORT = os.path.join(BASE, "output", "gate_diagnostic.json")

# Fallback lambda for the mass-matched uniform-KL control (plan-v2 T2).
#
# G0's 9.5% is the gate rate against the *fully SFT'd* released student. The pilot runs start
# from the pretrain adapter, where the student is far weaker and the teacher therefore wins
# much more often -- a 200-step smoke run measured 88% in its first quarter decaying to 24%
# by its last. Mass-matching against 0.095 would hand R1 much less KL mass than R2 and
# reintroduce the very confound T2 exists to remove. So R1's lambda is taken from R2's
# *realized* mean gate rate once R2 has actually run (--lambda_from_run r2_fgd), which the
# queue order makes possible: R0 -> R2 -> R1.
DEFAULT_MASS_MATCHED_LAMBDA = 0.095

FGD = {"distill_weight": 1.0, "distill_gate": "failure", "distill_margin": 0.0}


def run_specs(mass_matched_lambda):
    return {
        # priority order: the runs carrying the claim come first
        "r0_sac": ({}, "matched control: plain SAC SFT"),
        "r2_fgd": (dict(FGD), "proposed: failure-gated distillation"),
        "r1_uniform_matched": (
            {"distill_weight": mass_matched_lambda, "distill_gate": "none"},
            f"ablation: uniform KL at lambda={mass_matched_lambda:g} (matched KL mass)"),
        "r3_tga_fgd": (dict(FGD, encoder_query=True), "query-aware track: TGA + FGD"),
        "r3_tga": ({"encoder_query": True}, "query-aware track: TGA alone"),
        "r1a_uniform_lambda1": (
            {"distill_weight": 1.0, "distill_gate": "none"},
            "literature setting: uniform KL at lambda=1 (KV-Distill style)"),
    }


def realized_gate_rate(run_name):
    """Mean gate over a finished training run, from its per-example loss_info log."""
    path = os.path.join(SAC_HOME, "experiment", run_name, "output", "instruction_info.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        info = json.load(f)
    gates = [e["training_loss"]["gate"] for e in info
             if isinstance(e.get("training_loss"), dict) and "gate" in e["training_loss"]]
    if not gates:
        return None
    return sum(gates) / len(gates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda_from_run", default=None,
                    help="take the mass-matched lambda from this finished run's realized gate "
                         "rate instead of from G0 (e.g. r2_fgd)")
    ap.add_argument("--steps", type=int, default=5000, help="optimizer steps per run")
    ap.add_argument("--mid_checkpoint", type=int, default=2500,
                    help="also keep a named adapter at this step (plan-v2 T1)")
    ap.add_argument("--force", action="store_true", help="overwrite existing config.json")
    ap.add_argument("--only", nargs="*", default=None, help="only these run names")
    args = ap.parse_args()

    lam = DEFAULT_MASS_MATCHED_LAMBDA
    realized = realized_gate_rate(args.lambda_from_run) if args.lambda_from_run else None
    if realized is not None:
        lam = round(realized, 4)
        print(f"[configs] mass-matched lambda from {args.lambda_from_run}'s realized gate "
              f"rate: {lam:g}")
    elif os.path.exists(GATE_REPORT):
        with open(GATE_REPORT) as f:
            lam = json.load(f)["verdict"]["suggested_lambda_for_R1_mass_matched"]
        print(f"[configs] mass-matched lambda from G0 (converged student): {lam:g}")
        print("[configs] NOTE: re-generate r1_uniform_matched with --lambda_from_run r2_fgd "
              "once R2 has run, or its KL mass will not match R2's")
    else:
        print(f"[configs] no G0 report; using default lambda {lam:g}")

    with open(os.path.join(BASE, "config.json")) as f:
        base_cfg = json.load(f)
    src_adapter = os.path.join(BASE, "output", "adapter.pt")
    assert os.path.exists(src_adapter), f"missing pretrain adapter: {src_adapter}"

    specs = run_specs(lam)
    names = args.only or list(specs)
    for name in names:
        task_keys, note = specs[name]
        work_dir = os.path.join(SAC_HOME, "experiment", name)
        out_dir = os.path.join(work_dir, "output")
        os.makedirs(out_dir, exist_ok=True)

        cfg_path = os.path.join(work_dir, "config.json")
        if os.path.exists(cfg_path) and not args.force:
            print(f"[configs] {name}: config.json exists, skipping (use --force)")
        else:
            cfg = json.loads(json.dumps(base_cfg))  # deep copy
            cfg["sft_training_config"]["max_train_steps"] = args.steps
            if args.mid_checkpoint and args.mid_checkpoint < args.steps:
                cfg["sft_training_config"]["extra_checkpoint_steps"] = [args.mid_checkpoint]
            cfg["sft_task_config"].update(task_keys)
            cfg["pilot_note"] = note
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=4)
            print(f"[configs] {name}: {note}")
            print(f"           sft_task_config += {task_keys or '{}  (unchanged SAC)'}")

        # SFT starts from the released pretrain adapter; link rather than copy 27 MB per run
        dst_adapter = os.path.join(out_dir, "adapter.pt")
        if not os.path.exists(dst_adapter):
            try:
                os.symlink(os.path.relpath(src_adapter, out_dir), dst_adapter)
            except OSError:
                shutil.copy2(src_adapter, dst_adapter)
            print(f"           adapter.pt -> {os.path.relpath(src_adapter, out_dir)}")


if __name__ == "__main__":
    main()
