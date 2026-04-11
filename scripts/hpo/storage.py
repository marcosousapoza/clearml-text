import optuna

from data.cache import configure_cache_environment

from .config import HpoConfig, storage_uri, study_root


def delete_study(config: HpoConfig) -> None:
    cache_root = configure_cache_environment(config.cache_dir)
    study_dir = study_root(cache_root, config)
    uri = storage_uri(study_dir)
    if not (study_dir / "optuna.sqlite3").exists():
        print(f"[no-study-db] {study_dir / 'optuna.sqlite3'}")
        return

    summaries = optuna.study.get_all_study_summaries(storage=uri)
    if not any(summary.study_name == config.study_name for summary in summaries):
        print(f"[no-study] {config.study_name}")
        return

    optuna.delete_study(study_name=config.study_name, storage=uri)
    print(f"[deleted-study] {config.study_name}")
