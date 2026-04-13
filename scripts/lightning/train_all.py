from dataclasses import dataclass
from functools import partial
import multiprocessing as mp
import os
import subprocess
import sys
from typing import Any

from task import TASK_SPECS


@dataclass(frozen=True)
class TrainingJob:
    dataset: str
    task: str
    command: list[str]


@dataclass(frozen=True)
class TrainingResult:
    dataset: str
    task: str
    return_code: int


def main(argv: list[str] | None = None) -> None:
    import torch

    argv = sys.argv[1:] if argv is None else argv
    if argv:
        raise SystemExit("scripts.lightning.train_all does not accept arguments.")

    gpu_count = torch.cuda.device_count()
    gpu_ids = _visible_gpu_ids(gpu_count)
    parallelism = len(gpu_ids) if gpu_ids else 1
    accelerator = "gpu" if gpu_ids else "cpu"
    jobs = _build_jobs(accelerator)

    failures = _run_jobs(jobs, gpu_ids, parallelism)
    if failures:
        print("Failed tasks:")
        for dataset, task, return_code in failures:
            print(f"  {dataset}/{task}: exit code {return_code}")
        raise SystemExit(1)


def _build_jobs(accelerator: str) -> list[TrainingJob]:
    jobs = []
    for dataset, task, _task_cls in TASK_SPECS:
        jobs.append(
            TrainingJob(
                dataset=dataset,
                task=task,
                command=_build_command(dataset, task, accelerator),
            )
        )
    return jobs


def _run_jobs(
    jobs: list[TrainingJob],
    gpu_ids: list[str],
    parallelism: int,
) -> list[tuple[str, str, int]]:
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        gpu_queue = manager.Queue()
        for gpu_id in gpu_ids or [None]:
            gpu_queue.put(gpu_id)

        failures = []
        worker = partial(_run_job, gpu_queue=gpu_queue)
        with ctx.Pool(processes=parallelism) as pool:
            for result in pool.imap_unordered(worker, jobs, chunksize=1):
                if result.return_code != 0:
                    failures.append((result.dataset, result.task, result.return_code))
        return failures


def _run_job(job: TrainingJob, gpu_queue: Any) -> TrainingResult:
    gpu_id = gpu_queue.get()
    try:
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(_display_command(job.command, gpu_id), flush=True)
        completed = subprocess.run(job.command, env=env, check=False)
        return TrainingResult(job.dataset, job.task, completed.returncode)
    finally:
        gpu_queue.put(gpu_id)


def _build_command(
    dataset: str,
    task: str,
    accelerator: str,
) -> list[str]:
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
    ]
    if accelerator == "gpu":
        command.extend(["--devices", "1"])
    return command


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
