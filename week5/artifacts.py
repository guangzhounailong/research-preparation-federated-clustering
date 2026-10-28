"""运行配置、环境和完成记录；只复用同配置、同代码且内容完整的结果。"""

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def environment():
    import torch

    root = Path(__file__).resolve().parent
    sources = {
        str(path.relative_to(root)): file_hash(path)
        for path in sorted(root.rglob("*.py"))
        if "tests" not in path.parts
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ["numpy", "pandas", "scikit-learn", "torch", "matplotlib"]
        },
        "torch_threads": torch.get_num_threads(),
        "source_sha256": sources,
    }


def prepare_run(output_dir, config):
    config_data = json.loads(json.dumps(asdict(config)))
    current_environment = environment()
    if output_dir.exists() and any(output_dir.iterdir()):
        required = ["config.json", "environment.json", "run.json"]
        if not all((output_dir / name).is_file() for name in required):
            raise ValueError("输出目录包含未登记的结果，请指定新的 --output-dir。")
        if json.loads((output_dir / "config.json").read_text()) != config_data:
            raise ValueError("输出目录的实验配置不同，请指定新的 --output-dir。")
        if json.loads((output_dir / "environment.json").read_text()) != current_environment:
            raise ValueError("代码或运行环境已改变，请指定新的 --output-dir。")
        return json.loads((output_dir / "run.json").read_text())

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config_data)
    write_json(output_dir / "environment.json", current_environment)
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "stages": {}}
    write_json(output_dir / "run.json", manifest)
    return manifest


def stage_complete(output_dir, manifest, stage):
    record = manifest["stages"].get(str(stage), {})
    if record.get("status") != "complete":
        return False
    for name, expected in record["files"].items():
        path = output_dir / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Stage {stage} 的结果缺失或已修改：{name}；请使用新输出目录。")
    return True


def record_stage(output_dir, manifest, stage, status, error=None):
    record = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
    if status == "complete":
        stage_dir = output_dir / f"stage{stage}"
        files = sorted(path for path in stage_dir.rglob("*") if path.is_file())
        record["files"] = {str(path.relative_to(output_dir)): file_hash(path) for path in files}
    if error is not None:
        record["error"] = str(error)
    manifest["stages"][str(stage)] = record
    write_json(output_dir / "run.json", manifest)
