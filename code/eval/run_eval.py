# code/eval/run_eval.py

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def _as_csv(x) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return ",".join(str(i).strip() for i in x if str(i).strip())
    return ",".join([t.strip() for t in str(x).split(",") if t.strip()])


def _resolve_under(base: Path, p) -> Path:
    """Resolve path p under base unless it is absolute."""
    if p is None:
        return None
    pp = Path(str(p))
    if pp.is_absolute():
        return pp
    return (base / pp).resolve()


def _run(cmd, log_path: Path, env=None, dry_run=False):
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n[RUN] {cmd_str}")
    print(f"[LOG] {log_path}")

    if dry_run:
        return 0

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(cmd_str + "\n\n")
        f.flush()
        p = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
    return p.returncode


def build_parser():
    ap = argparse.ArgumentParser(
        description="Unified evaluation runner driven by eval_config.yaml (with optional CLI overrides)."
    )

    ap.add_argument("--config", type=str, default="eval_config.yaml",
                    help="Path to YAML config (relative to this file if not absolute).")

    ap.add_argument("--tasks", type=str, default="",
                    help="Override tasks: comma list (eurosat,levircd,oscd,s2looking,timesen2crop) or 'all'.")

    ap.add_argument("--python", type=str, default=sys.executable,
                    help="Python executable to use (default: current).")

    # Optional path overrides (if not provided, config is used)
    ap.add_argument("--data_root", type=str, default="")
    ap.add_argument("--models_root", type=str, default="")
    ap.add_argument("--teacher_dir", type=str, default="")
    ap.add_argument("--teacher_ckpt", type=str, default="")
    ap.add_argument("--student_ckpts", type=str, default="",
                    help="Override student ckpts as comma list.")

    ap.add_argument("--levir_root", type=str, default="")
    ap.add_argument("--oscd_root", type=str, default="")
    ap.add_argument("--s2looking_root", type=str, default="")
    ap.add_argument("--timesen2crop_cache_dir", type=str, default="")

    # Optional hyperparam overrides
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num_frames", type=int, default=None)

    ap.add_argument("--crop", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)

    # Student architecture overrides
    ap.add_argument("--student_embed_dim", type=int, default=None)
    ap.add_argument("--student_depth", type=int, default=None)
    ap.add_argument("--student_heads", type=int, default=None)
    ap.add_argument("--K_reg", type=int, default=None)

    # OSCD band overrides
    ap.add_argument("--band_order", type=str, default="")
    ap.add_argument("--use_bands", type=str, default="")

    # S2Looking overrides
    ap.add_argument("--s2looking_label", type=str, default="",
                    choices=["", "label", "label1", "label2"])
    ap.add_argument("--posrate_n", type=int, default=None)

    # TimeSen2Crop overrides
    ap.add_argument("--timesen2crop_fold", type=int, default=None)
    ap.add_argument("--timesen2crop_train_n", type=int, default=None)
    ap.add_argument("--timesen2crop_test_n", type=int, default=None)
    ap.add_argument("--timesen2crop_batch_size", type=int, default=None)
    ap.add_argument("--timesen2crop_img_size", type=int, default=None)
    ap.add_argument("--timesen2crop_days", type=str, default="")
    ap.add_argument("--timesen2crop_band_idx", type=str, default="")
    ap.add_argument("--timesen2crop_use", type=str, default="",
                    choices=["", "cls", "pm", "cat"])
    ap.add_argument("--probe_epochs", type=int, default=None)
    ap.add_argument("--probe_lr", type=float, default=None)
    ap.add_argument("--probe_wd", type=float, default=None)

    # Runner behavior
    ap.add_argument("--log_dir", type=str, default="",
                    help="Override log_dir (default comes from config).")
    ap.add_argument("--continue_on_error", action="store_true",
                    help="If set, keep going even if a task fails (overrides config false).")
    ap.add_argument("--dry_run", action="store_true")
    return ap


def main():
    eval_dir = Path(__file__).resolve().parent          # project_root/code/eval
    code_root = eval_dir.parent                         # project_root/code
    project_root = code_root.parent                     # project_root

    args = build_parser().parse_args()

    # ---- Load config ----
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (eval_dir / cfg_path).resolve()

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- Resolve roots ----
    paths_cfg = cfg.get("paths", {})

    data_root = _resolve_under(project_root, args.data_root or paths_cfg.get("data_root", "../data"))
    models_root = _resolve_under(project_root, args.models_root or paths_cfg.get("models_root", "../models"))

    # Teacher + student live under models_root
    teacher_dir = _resolve_under(models_root, args.teacher_dir or paths_cfg.get("teacher_dir"))
    teacher_ckpt = _resolve_under(models_root, args.teacher_ckpt or paths_cfg.get("teacher_ckpt"))

    # student_ckpts: config list OR CLI comma list
    if args.student_ckpts.strip():
        student_ckpts_csv = _as_csv(args.student_ckpts)
    else:
        student_list = paths_cfg.get("student_ckpts", [])
        # resolve each under models_root
        student_ckpts_csv = ",".join(str(_resolve_under(models_root, p)) for p in student_list)

    # ---- Tasks ----
    runner_cfg = cfg.get("runner", {})
    if args.tasks.strip():
        tasks_str = args.tasks.strip().lower()
        if tasks_str == "all":
            task_list = ["eurosat", "levircd", "oscd", "s2looking", "timesen2crop"]
        else:
            task_list = [t.strip() for t in tasks_str.split(",") if t.strip()]
    else:
        task_list = [t.strip().lower() for t in runner_cfg.get("tasks", [])]
        if not task_list:
            task_list = ["eurosat", "levircd", "oscd", "s2looking", "timesen2crop"]

    # ---- Common knobs ----
    model_cfg = cfg.get("model", {})
    common_cfg = cfg.get("common", {})
    cd_cfg = cfg.get("change_detection", {})

    seed = args.seed if args.seed is not None else int(common_cfg.get("seed", 42))
    num_frames = args.num_frames if args.num_frames is not None else int(common_cfg.get("num_frames", 3))

    crop = args.crop if args.crop is not None else int(cd_cfg.get("crop", 256))
    batch_size = args.batch_size if args.batch_size is not None else int(cd_cfg.get("batch_size", 8))
    epochs = args.epochs if args.epochs is not None else int(cd_cfg.get("epochs", 5))
    lr = args.lr if args.lr is not None else float(cd_cfg.get("lr", 1e-3))

    student_embed_dim = args.student_embed_dim if args.student_embed_dim is not None else model_cfg.get("student_embed_dim", None)
    student_depth = args.student_depth if args.student_depth is not None else model_cfg.get("student_depth", None)
    student_heads = args.student_heads if args.student_heads is not None else model_cfg.get("student_heads", None)
    K_reg = args.K_reg if args.K_reg is not None else model_cfg.get("K_reg", None)

    # ---- Dataset paths ----
    ds_cfg = cfg.get("datasets", {})
    levir_root = _resolve_under(data_root, args.levir_root or ds_cfg.get("levircd_root", "LEVIRCD"))
    oscd_root = _resolve_under(data_root, args.oscd_root or ds_cfg.get("oscd_root", "oscd"))
    s2looking_root = _resolve_under(data_root, args.s2looking_root or ds_cfg.get("s2looking_root", "S2Looking"))
    timesen_cache = _resolve_under(data_root, args.timesen2crop_cache_dir or ds_cfg.get("timesen2crop_cache_dir", "timesen2crop_cache"))

    # ---- OSCD bands ----
    oscd_cfg = cfg.get("oscd", {})
    band_order = _as_csv(args.band_order or oscd_cfg.get("band_order", "B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12"))
    use_bands = _as_csv(args.use_bands or oscd_cfg.get("use_bands", "B02,B03,B04,B08,B11,B12"))

    # ---- S2Looking ----
    s2_cfg = cfg.get("s2looking", {})
    s2_label = args.s2looking_label or str(s2_cfg.get("use_label", "label"))
    posrate_n = args.posrate_n if args.posrate_n is not None else int(s2_cfg.get("posrate_n", 200))

    # ---- TimeSen2Crop ----
    tsc_cfg = cfg.get("timesen2crop", {})
    t_fold = args.timesen2crop_fold if args.timesen2crop_fold is not None else int(tsc_cfg.get("fold", 0))
    t_train_n = args.timesen2crop_train_n if args.timesen2crop_train_n is not None else int(tsc_cfg.get("train_n", 50000))
    t_test_n = args.timesen2crop_test_n if args.timesen2crop_test_n is not None else int(tsc_cfg.get("test_n", 20000))
    t_bs = args.timesen2crop_batch_size if args.timesen2crop_batch_size is not None else int(tsc_cfg.get("batch_size", 64))
    t_img = args.timesen2crop_img_size if args.timesen2crop_img_size is not None else int(tsc_cfg.get("img_size", 64))
    t_days = _as_csv(args.timesen2crop_days or tsc_cfg.get("days", "60,180,300"))
    t_band_idx = _as_csv(args.timesen2crop_band_idx or tsc_cfg.get("band_idx", "0,1,2,6,7,8"))
    t_use = args.timesen2crop_use or str(tsc_cfg.get("use", "cls"))
    probe_epochs = args.probe_epochs if args.probe_epochs is not None else int(tsc_cfg.get("probe_epochs", 8))
    probe_lr = args.probe_lr if args.probe_lr is not None else float(tsc_cfg.get("probe_lr", 1e-3))
    probe_wd = args.probe_wd if args.probe_wd is not None else float(tsc_cfg.get("probe_wd", 1e-4))

    # ---- Logging + env ----
    log_dir_name = args.log_dir or runner_cfg.get("log_dir", "eval_logs")
    log_root = _ensure_dir(eval_dir / log_dir_name / _ts())

    continue_on_error = bool(args.continue_on_error) or bool(runner_cfg.get("continue_on_error", False))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(code_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    print(f"project_root: {project_root}")
    print(f"code_root:    {code_root}")
    print(f"eval_dir:     {eval_dir}")
    print(f"data_root:    {data_root}")
    print(f"models_root:  {models_root}")
    print(f"log_root:     {log_root}")
    print(f"tasks:        {task_list}")
    print(f"teacher_dir:  {teacher_dir}")
    print(f"teacher_ckpt: {teacher_ckpt}")
    print(f"student_ckpts:{student_ckpts_csv}")

    results = []
    failures = 0

    def _add_student_arch(cmd):
        if student_embed_dim is not None:
            cmd += ["--student_embed_dim", str(student_embed_dim)]
        if student_depth is not None:
            cmd += ["--student_depth", str(student_depth)]
        if student_heads is not None:
            cmd += ["--student_heads", str(student_heads)]
        if K_reg is not None:
            cmd += ["--K_reg", str(K_reg)]
        return cmd

    def run_task(name, cmd):
        nonlocal failures
        log_path = log_root / f"{name}.log"
        rc = _run(cmd, log_path, env=env, dry_run=args.dry_run)
        results.append((name, rc, str(log_path)))
        if rc != 0:
            failures += 1
            print(f"[FAIL] {name} exit code {rc}")
            if not continue_on_error:
                print("Stopping because continue_on_error is false.")
                sys.exit(rc)
        else:
            print(f"[OK] {name}")

    # ---- EuroSAT ----
    if "eurosat" in task_list:
        # (still hardcoded inside eval_eurosat.py unless you add argparse there)
        cmd = [args.python, str(eval_dir / "eval_eurosat.py")]
        run_task("eurosat", cmd)

    # ---- LEVIR-CD ----
    if "levircd" in task_list:
        if not levir_root.exists() and not args.dry_run:
            raise SystemExit(f"LEVIRCD root not found: {levir_root}")

        cmd = [
            args.python, str(eval_dir / "eval_levircd.py"),
            "--levir_root", str(levir_root),
            "--teacher_dir", str(teacher_dir),
            "--teacher_ckpt", str(teacher_ckpt),
            "--student_ckpts", student_ckpts_csv,
            "--crop", str(crop),
            "--batch_size", str(batch_size),
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--num_frames", str(num_frames),
            "--seed", str(seed),
        ]
        cmd = _add_student_arch(cmd)
        run_task("levircd", cmd)

    # ---- OSCD ----
    if "oscd" in task_list:
        if not oscd_root.exists() and not args.dry_run:
            raise SystemExit(f"OSCD root not found: {oscd_root}")

        cmd = [
            args.python, str(eval_dir / "eval_oscd.py"),
            "--oscd_root", str(oscd_root),
            "--teacher_dir", str(teacher_dir),
            "--teacher_ckpt", str(teacher_ckpt),
            "--student_ckpts", student_ckpts_csv,
            "--crop", str(crop),
            "--batch_size", str(batch_size),
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--num_frames", str(num_frames),
            "--band_order", band_order,
            "--use_bands", use_bands,
            "--seed", str(seed),
        ]
        cmd = _add_student_arch(cmd)
        run_task("oscd", cmd)

    # ---- S2Looking ----
    if "s2looking" in task_list:
        if not s2looking_root.exists() and not args.dry_run:
            raise SystemExit(f"S2Looking root not found: {s2looking_root}")

        cmd = [
            args.python, str(eval_dir / "eval_s2looking.py"),
            "--data_root", str(s2looking_root),
            "--teacher_dir", str(teacher_dir),
            "--teacher_ckpt", str(teacher_ckpt),
            "--student_ckpts", student_ckpts_csv,
            "--crop", str(crop),
            "--batch_size", str(batch_size),
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--num_frames", str(num_frames),
            "--use_label", s2_label,
            "--posrate_n", str(posrate_n),
            "--seed", str(seed),
        ]
        cmd = _add_student_arch(cmd)
        run_task("s2looking", cmd)

    # ---- TimeSen2Crop ----
    if "timesen2crop" in task_list:
        cmd = [
            args.python, str(eval_dir / "eval_timesen2crop.py"),
            "--cache_dir", str(timesen_cache),
            "--fold", str(t_fold),
            "--train_n", str(t_train_n),
            "--test_n", str(t_test_n),
            "--batch_size", str(t_bs),
            "--img_size", str(t_img),
            "--days", t_days,
            "--band_idx", t_band_idx,
            "--use", t_use,
            "--teacher_dir", str(teacher_dir),
            "--teacher_ckpt", str(teacher_ckpt),
            "--student_ckpts", student_ckpts_csv,
            "--probe_epochs", str(probe_epochs),
            "--probe_lr", str(probe_lr),
            "--probe_wd", str(probe_wd),
            "--seed", str(seed),
        ]
        cmd = _add_student_arch(cmd)
        run_task("timesen2crop", cmd)

    print("\n=== Summary ===")
    for name, rc, logp in results:
        status = "OK" if rc == 0 else "FAIL"
        print(f"{name:12s} {status:4s}  rc={rc}  log={logp}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
