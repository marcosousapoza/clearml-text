import optuna

from data.cache import configure_cache_environment

from .config import HpoConfig, optuna_storage, storage_artifact_path, study_root


def delete_study(config: HpoConfig) -> None:
    cache_root = configure_cache_environment(config.cache_dir)
    study_dir = study_root(cache_root, config)
    artifact_path = storage_artifact_path(study_dir, config.storage_backend)
    if not artifact_path.exists():
        print(f"[no-study-db] {artifact_path}")
        return

    storage = optuna_storage(study_dir, config.storage_backend)
    summaries = optuna.study.get_all_study_summaries(storage=storage)
    if not any(summary.study_name == config.study_name for summary in summaries):
        print(f"[no-study] {config.study_name}")
        return

    optuna.delete_study(study_name=config.study_name, storage=storage)
    print(f"[deleted-study] {config.study_name}")
