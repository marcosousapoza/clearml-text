from dataclasses import dataclass
from functools import partial
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from data.cache import get_cache_root
from task import TASK_SPECS


TRAIN_ALL_SEEDS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class TrainingJob:
    dataset: str
    task: str
    seed: int
    command: list[str]
    log_path: Path


@dataclass(frozen=True)
class TrainingResult:
    dataset: str
    task: str
    seed: int
    return_code: int


def main(argv: list[str] | None = None) -> None:
    import argparse
    import torch

    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Train all registered tasks with PyTorch Lightning.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset name to train. Repeat to select multiple datasets. Defaults to all datasets.",
    )
    parser.add_argument("--flatten", action="store_true", help="Flatten databases to each task's object types.")
    parser.add_argument("--accelerator", type=str, default=None, help="Accelerator override (e.g. cpu, gpu). Defaults to gpu if available, else cpu.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs. Defaults to the scripts.lightning default.")
    parser.add_argument("--wandb", action="store_true", help="Also log metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="ocel-ocp", help="W&B project name.")
    parser.add_argument("--jobs-per-gpu", type=int, default=1, help="Number of training jobs to run in parallel per GPU.")
    args = parser.parse_args(argv)

    gpu_count = torch.cuda.device_count()
    gpu_ids = _visible_gpu_ids(gpu_count)
    parallelism = len(gpu_ids) * args.jobs_per_gpu if gpu_ids else 1
    accelerator = args.accelerator or ("gpu" if gpu_ids else "cpu")
    if accelerator == "cpu":
        gpu_ids = []
    jobs = _build_jobs(accelerator, set(args.dataset), args.flatten, args.epochs, args.wandb, args.wandb_project)

    failures = _run_jobs(jobs, gpu_ids, parallelism, args.jobs_per_gpu)
    if failures:
        print("Failed tasks:")
        for dataset, task, seed, return_code in failures:
            print(f"  {dataset}/{task} seed={seed}: exit code {return_code}")
        raise SystemExit(1)


def _build_jobs(accelerator: str, datasets: set[str], flatten: bool, epochs: int | None = None, wandb: bool = False, wandb_project: str = "ocel-ocp") -> list[TrainingJob]:
    jobs = []
    available_datasets = {dataset for dataset, _task, _task_cls in TASK_SPECS}
    unknown_datasets = datasets - available_datasets
    if unknown_datasets:
        raise ValueError(
            "Unknown dataset(s): "
            f"{', '.join(sorted(unknown_datasets))}. "
            f"Available datasets: {', '.join(sorted(available_datasets))}."
        )

    for dataset, task, _task_cls in TASK_SPECS:
        if datasets and dataset not in datasets:
            continue
        for seed in TRAIN_ALL_SEEDS:
            command, log_path = _build_command(dataset, task, accelerator, seed, flatten, epochs, wandb, wandb_project)
            jobs.append(
                TrainingJob(
                    dataset=dataset,
                    task=task,
                    seed=seed,
                    command=command,
                    log_path=log_path,
                )
            )
    return jobs


def _run_jobs(
    jobs: list[TrainingJob],
    gpu_ids: list[str],
    parallelism: int,
    jobs_per_gpu: int = 1,
) -> list[tuple[str, str, int, int]]:
    # Capture the parent's environment (including .env variables loaded by load_env())
    # before spawning workers, because spawn workers start with the OS environment and
    # do not inherit changes made to os.environ in the parent process.
    parent_env = os.environ.copy()

    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        gpu_queue = manager.Queue()
        for gpu_id in gpu_ids or [None]:
            for _ in range(jobs_per_gpu):
                gpu_queue.put(gpu_id)

        failures = []
        worker = partial(_run_job, gpu_queue=gpu_queue, parent_env=parent_env)
        with ctx.Pool(processes=parallelism) as pool:
            for result in pool.imap_unordered(worker, jobs, chunksize=1):
                if result.return_code != 0:
                    failures.append((result.dataset, result.task, result.seed, result.return_code))
        return failures


def _run_job(job: TrainingJob, gpu_queue: Any, parent_env: dict[str, str] | None = None) -> TrainingResult:
    gpu_id = gpu_queue.get()
    try:
        env = (parent_env or os.environ).copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        print(_display_command(job.command, gpu_id), f"  -> {job.log_path}", flush=True)
        with open(job.log_path, "w") as log_file:
            completed = subprocess.run(job.command, env=env, check=False, stdout=log_file, stderr=log_file)
        return TrainingResult(job.dataset, job.task, job.seed, completed.returncode)
    finally:
        gpu_queue.put(gpu_id)


def _build_command(
    dataset: str,
    task: str,
    accelerator: str,
    seed: int,
    flatten: bool,
    epochs: int | None = None,
    wandb: bool = False,
    wandb_project: str = "ocel-ocp",
) -> tuple[list[str], Path]:
    run_name = f"{dataset}_{task}{'_flat' if flatten else ''}"
    root_dir = get_cache_root() / run_name / "lightning" / f"seed_{seed}"
    log_path = get_cache_root() / ".logs" / run_name / f"seed_{seed}.log"
    command = [
        sys.executable,
        "-m",
        "scripts.lightning",
        "--dataset",
        dataset,
        "--task",
        task,
        "--accelerator",
        accelerator,
        "--seed",
        str(seed),
        "--default-root-dir",
        str(root_dir),
    ]
    if epochs is not None:
        command.extend(["--epochs", str(epochs)])
    if flatten:
        command.append("--flatten")
    if accelerator == "gpu":
        command.extend(["--devices", "1"])
    if wandb:
        command.extend(["--wandb", "--wandb_project", wandb_project])
    return command, log_path


def _display_command(command: list[str], gpu_id: str | None) -> str:
    prefix = f"CUDA_VISIBLE_DEVICES={gpu_id} " if gpu_id is not None else ""
    return prefix + " ".join(command)


def _visible_gpu_ids(gpu_count: int) -> list[str]:
    if gpu_count <= 0:
        return []

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        gpu_ids = [gpu_id.strip() for gpu_id in visible_devices.split(",") if gpu_id.strip()]
        if gpu_ids:
            return gpu_ids[:gpu_count]

    return [str(gpu_id) for gpu_id in range(gpu_count)]


if __name__ == "__main__":
    main()
