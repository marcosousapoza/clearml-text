"""Manage ClearML + Optuna HPO runs."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import optuna
from clearml import Task
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
# ruff: noqa: E402

from scripts.utils.config import ConfigError, ExperimentConfig
from scripts.utils.experiments import PROJECTIONS, TASK_KINDS, expand_experiments
from src.model.config.parser import parse_search_space

CLEARML_PROJECT = "ocel-ocp"
STATE_DIR = ROOT_DIR / "out" / "hpo" / "manager"
OPTUNA_STORAGE_PATH = ROOT_DIR / "out" / "hpo" / "optuna.sqlite3"
BOOTSTRAP_WAIT_TIMEOUT_S = 60 * 60 * 8
BOOTSTRAP_POLL_INTERVAL_S = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start, stop, and inspect HPO runs.")
    parser.add_argument("action", choices=("run", "start", "stop", "restart", "status", "delete"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task-kind", choices=TASK_KINDS, default=None)
    parser.add_argument("--projection", choices=PROJECTIONS, default=None)
    parser.add_argument("--object-types", nargs="+", default=None)
    parser.add_argument("--lookback", default=None)
    parser.add_argument("--horizon", default=None)
    parser.add_argument("--negative-ratio", type=float, default=None)
    parser.add_argument("--queue", default=None)
    parser.add_argument("--hpo-queue", default=None)
    parser.add_argument("--term-timeout", type=float, default=15.0)
    return parser.parse_args()


def _task_sort_key(task: Task) -> str:
    data = getattr(task, "data", None)
    for attr in ("last_update", "completed", "started", "created"):
        value = getattr(data, attr, None) if data is not None else None
        if value:
            return str(value)
    return ""


def _get_most_recent_task(*, task_name: str, project_name: str) -> Task | None:
    tasks = Task.get_tasks(
        task_name=task_name,
        project_name=project_name,
        allow_archived=False,
    )
    if not tasks:
        return None
    return max(tasks, key=_task_sort_key)


def _load_or_create_study(exp: ExperimentConfig) -> optuna.Study:
    OPTUNA_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    study_name = f"{CLEARML_PROJECT}:{exp['name']}"
    storage_uri = f"sqlite:///{OPTUNA_STORAGE_PATH.resolve()}"
    direction = "maximize" if str(exp["hpo"]["objective_sign"]).lower() == "max" else "minimize"
    print(f"[hpo] using study name={study_name!r} storage={storage_uri}")
    return optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        direction=direction,
        load_if_exists=True,
    )


def _study_name(exp: ExperimentConfig) -> str:
    return f"{CLEARML_PROJECT}:{exp['name']}"


def _storage_uri() -> str:
    return f"sqlite:///{OPTUNA_STORAGE_PATH.resolve()}"


def _selection_command(script_name: str, exp: ExperimentConfig) -> list[str]:
    command = [
        "python",
        script_name,
        "--dataset",
        exp["dataset"],
        "--task-kind",
        exp["task"]["task_kind"],
        "--projection",
        exp["projection"],
    ]
    object_types = exp["task"]["object_types"]
    if object_types:
        command.extend(["--object-types", *object_types])
    if exp["task"].get("lookback") is not None:
        command.extend(["--lookback", str(exp["task"]["lookback"])])
    if exp["task"].get("horizon") is not None:
        command.extend(["--horizon", str(exp["task"]["horizon"])])
    if exp["task"].get("negative_ratio") is not None:
        command.extend(["--negative-ratio", str(exp["task"]["negative_ratio"])])
    if exp["run"].get("queue") is not None:
        command.extend(["--queue", str(exp["run"]["queue"])])
    return command


def _bootstrap_base_task(exp: ExperimentConfig) -> None:
    command = _selection_command("scripts/gnn_train.py", exp)
    print(f"[hpo] bootstrapping base task for {exp['name']}")
    subprocess.run(command, cwd=ROOT_DIR, env=dict(os.environ), check=True)


def _task_has_metric(task: Task, title: str, series: str) -> bool:
    metrics = task.get_last_scalar_metrics()
    return title in metrics and series in metrics[title]


def _wait_for_bootstrap_task(exp: ExperimentConfig, previous_task_id: str | None) -> Task:
    objective_title = exp["hpo"]["objective_title"]
    objective_series = exp["hpo"]["objective_series"]
    deadline = time.monotonic() + BOOTSTRAP_WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        task = _get_most_recent_task(task_name=exp["name"], project_name=CLEARML_PROJECT)
        if task is None or task.id == previous_task_id:
            time.sleep(BOOTSTRAP_POLL_INTERVAL_S)
            continue

        task.reload()
        status = str(getattr(task.data, "status", "") or "").lower()
        if _task_has_metric(task, objective_title, objective_series):
            print(
                f"[hpo] bootstrap task ready id={task.id} "
                f"status={status or '<unknown>'} metric={objective_title}/{objective_series}"
            )
            return task
        if status in {"failed", "stopped"}:
            raise SystemExit(
                f"[hpo] bootstrap task {task.id} ended with status={status!r} "
                f"before reporting {objective_title}/{objective_series}"
            )
        print(
            f"[hpo] waiting for bootstrap task id={task.id} "
            f"status={status or '<unknown>'} metric={objective_title}/{objective_series}"
        )
        time.sleep(BOOTSTRAP_POLL_INTERVAL_S)
    raise SystemExit(
        f"[hpo] timed out waiting for bootstrap task to report "
        f"{objective_title}/{objective_series}"
    )


def _pid_path(exp: ExperimentConfig) -> Path:
    return STATE_DIR / f"{exp['name']}.pid"


def _meta_path(exp: ExperimentConfig) -> Path:
    return STATE_DIR / f"{exp['name']}.json"


def _log_path(exp: ExperimentConfig) -> Path:
    return STATE_DIR / f"{exp['name']}.log"


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="ascii").strip())
    except ValueError:
        return None


def _is_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_state(exp: ExperimentConfig, pid: int, command: list[str]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _pid_path(exp).write_text(f"{pid}\n", encoding="ascii")
    _meta_path(exp).write_text(
        json.dumps({
            "pid": pid,
            "name": exp["name"],
            "dataset": exp["dataset"],
            "task_kind": exp["task"]["task_kind"],
            "projection": exp["projection"],
            "log": str(_log_path(exp).relative_to(ROOT_DIR)),
            "command": command,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _wait_for_exit(pid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _is_running(pid):
            return True
        time.sleep(0.25)
    return not _is_running(pid)


def _stop_process(exp: ExperimentConfig, timeout_s: float) -> None:
    pid = _read_pid(_pid_path(exp))
    if not _is_running(pid):
        print(f"[not-running] {exp['name']}")
        if _pid_path(exp).exists():
            _pid_path(exp).unlink()
        return
    assert pid is not None
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if not _wait_for_exit(pid, timeout_s):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _wait_for_exit(pid, 2.0)
    if _pid_path(exp).exists():
        _pid_path(exp).unlink()
    print(f"[stopped] {exp['name']} pid={pid}")


def _delete_process_state(exp: ExperimentConfig) -> None:
    for path in (_pid_path(exp), _meta_path(exp), _log_path(exp)):
        if path.exists():
            path.unlink()


def _delete_study(exp: ExperimentConfig) -> None:
    if not OPTUNA_STORAGE_PATH.exists():
        print(f"[no-study-db] {OPTUNA_STORAGE_PATH.relative_to(ROOT_DIR)}")
        return
    try:
        summaries = optuna.study.get_all_study_summaries(storage=_storage_uri())
    except KeyError:
        summaries = []
    study_name = _study_name(exp)
    if not any(summary.study_name == study_name for summary in summaries):
        print(f"[no-study] {study_name}")
        return
    optuna.delete_study(study_name=study_name, storage=_storage_uri())
    print(f"[deleted-study] {study_name}")


def _status_process(exp: ExperimentConfig) -> None:
    pid = _read_pid(_pid_path(exp))
    state = "running" if _is_running(pid) else "stopped"
    suffix = f" pid={pid}" if pid is not None else ""
    print(f"[{state}] {exp['name']}{suffix} log={_log_path(exp).relative_to(ROOT_DIR)}")


def _start_process(exp: ExperimentConfig) -> None:
    pid = _read_pid(_pid_path(exp))
    if _is_running(pid):
        print(f"[already-running] {exp['name']} pid={pid}")
        return
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    command = _selection_command("scripts/hpo.py", exp)
    command.insert(4, "run")
    with _log_path(exp).open("ab") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _write_state(exp, proc.pid, command)
    print(f"[started] {exp['name']} pid={proc.pid} log={_log_path(exp).relative_to(ROOT_DIR)}")


def _resolve_experiments(args: argparse.Namespace) -> list[ExperimentConfig]:
    return expand_experiments(
        dataset=args.dataset,
        task_kind=args.task_kind,
        projection=args.projection,
        object_types=tuple(args.object_types or ()),
        lookback=args.lookback,
        horizon=args.horizon,
        negative_ratio=args.negative_ratio,
        queue=args.queue,
        hpo_queue=args.hpo_queue,
    )


def _run_hpo(exp: ExperimentConfig) -> None:
    if exp["search_space"] is None:
        raise SystemExit("[hpo] search_space must be defined for HPO")
    if exp["hpo"] is None:
        raise SystemExit("[hpo] hpo config must be defined for HPO")

    space = parse_search_space(exp["search_space"])
    if not space:
        raise SystemExit("[hpo] search_space parsed empty")

    base_task = _get_most_recent_task(task_name=exp["name"], project_name=CLEARML_PROJECT)
    if base_task is None:
        previous_task_id = None
    else:
        previous_task_id = base_task.id
        if not _task_has_metric(base_task, exp["hpo"]["objective_title"], exp["hpo"]["objective_series"]):
            previous_task_id = base_task.id
            base_task = None
    if base_task is None:
        _bootstrap_base_task(exp)
        base_task = _wait_for_bootstrap_task(exp, previous_task_id)
    print(f"[hpo] selected base task id={base_task.id} name={base_task.name}")

    study = _load_or_create_study(exp)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
    )

    optimizer = HyperParameterOptimizer(
        base_task_id=base_task.id,
        hyper_parameters=space,
        objective_metric_title=exp["hpo"]["objective_title"],
        objective_metric_series=exp["hpo"]["objective_series"],
        objective_metric_sign=exp["hpo"]["objective_sign"],
        max_number_of_concurrent_tasks=exp["hpo"]["parallel"],
        total_max_jobs=exp["hpo"]["total_jobs"],
        max_iteration_per_job=exp["run"]["epochs"],
        optimizer_class=OptimizerOptuna,
        execution_queue=exp["hpo"]["queue"],
        project_name=CLEARML_PROJECT,
        continue_previous_study=study,
        optuna_pruner=pruner,
    )

    optimizer.start()
    try:
        optimizer.wait()
    finally:
        optimizer.stop()


def main() -> None:
    args = parse_args()
    try:
        experiments = _resolve_experiments(args)
    except (ConfigError, ValueError) as exc:
        raise SystemExit(f"[hpo] {exc}") from exc

    if args.action == "run":
        if len(experiments) != 1:
            raise SystemExit("[hpo] 'run' requires exactly one experiment selection.")
        _run_hpo(experiments[0])
        return

    if args.action == "start":
        for exp in experiments:
            _start_process(exp)
        return

    if args.action == "stop":
        for exp in experiments:
            _stop_process(exp, args.term_timeout)
        return

    if args.action == "restart":
        for exp in experiments:
            _stop_process(exp, args.term_timeout)
            _start_process(exp)
        return

    if args.action == "status":
        for exp in experiments:
            _status_process(exp)
        return

    if args.action == "delete":
        for exp in experiments:
            _stop_process(exp, args.term_timeout)
            _delete_process_state(exp)
            _delete_study(exp)
        return

    raise SystemExit(f"[hpo] Unsupported action: {args.action}")


if __name__ == "__main__":
    main()
