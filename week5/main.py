"""统一实验入口：python -m week5.main --stage all。"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from importlib import import_module
import math
from pathlib import Path

from .config import DEFAULT_CONFIG


STAGES = {
    1: "centralized",
    2: "partition",
    3: "local_only",
    4: "federated_ae",
    5: "federated_vae",
    6: "privacy",
    7: "ablation",
    8: "visualization",
    9: "comparison",
}
# 这些依赖只用于复用结果，不改变各阶段的训练设置。
DEPENDENCIES = {4: (3,), 5: (4,), 8: (1, 4, 7), 9: (1, 3, 4, 5, 7)}


def stage_order(requested):
    order = []

    def add(stage):
        for dependency in DEPENDENCIES.get(stage, ()):
            add(dependency)
        if stage not in order:
            order.append(stage)

    for stage in requested:
        add(stage)
    return order


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Digits 联邦表示学习：按 Stage 运行，并自动补齐所需阶段。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", nargs="+", default=["all"],
                        help="all 或阶段编号，例如 --stage 1 3 4")
    parser.add_argument("--output-dir", type=Path,
                        help="同目录可接着运行其他阶段；默认创建 UTC 时间命名的新目录")
    options = [
        ("--epochs", "epochs", int, "集中式与 Local-only AE/VAE 训练次数"),
        ("--rounds", "communication_rounds", int, "联邦通信轮数"),
        ("--local-epochs", "local_epochs", int, "每轮客户端训练次数"),
        ("--batch-size", "batch_size", int, "批大小"),
        ("--latent-dim", "latent_dim", int, "AE / VAE 表示维度"),
        ("--pca-dim", "pca_dim", int, "PCA 表示维度"),
        ("--learning-rate", "learning_rate", float, "Adam 学习率"),
        ("--beta", "beta", float, "VAE 的 KL 损失权重"),
        ("--split-seed", "split_seed", int, "训练/测试集划分种子"),
        ("--partition-seed", "partition_seed", int, "客户端划分种子"),
        ("--num-clients", "num_clients", int, "客户端数，需能整除十个类别"),
        ("--clip-norm", "clip_norm", float, "Stage 6 更新裁剪阈值"),
        ("--ablation-clip-norm", "ablation_clip_norm", float, "Stage 7–9 裁剪阈值"),
        ("--ablation-noise-std", "ablation_noise_std", float, "Stage 7–9 固定噪声标准差"),
    ]
    for flag, name, kind, help_text in options:
        parser.add_argument(flag, dest=name, type=kind, default=getattr(DEFAULT_CONFIG, name),
                            help=help_text)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_CONFIG.seeds,
                        help="训练随机种子")
    parser.add_argument("--noise-multipliers", nargs="+", type=float,
                        default=DEFAULT_CONFIG.noise_multipliers,
                        help="Stage 6 噪声倍数；实际标准差为倍数乘裁剪阈值")
    args = parser.parse_args(argv)
    try:
        if args.stage == ["all"]:
            requested = list(STAGES)
        else:
            requested = [int(stage) for stage in args.stage]
            if not all(stage in STAGES for stage in requested):
                raise ValueError
    except ValueError:
        parser.error("--stage 应为 all，或 1–9 的阶段编号。")

    overrides = {name: getattr(args, name) for _, name, _, _ in options}
    overrides["seeds"] = tuple(args.seeds)
    overrides["noise_multipliers"] = tuple(args.noise_multipliers)
    config = replace(DEFAULT_CONFIG, **overrides)
    positive = ["epochs", "communication_rounds", "local_epochs", "batch_size", "latent_dim",
                "pca_dim", "learning_rate", "num_clients", "clip_norm", "ablation_clip_norm"]
    if any(not math.isfinite(getattr(config, name)) or getattr(config, name) <= 0 for name in positive):
        parser.error("训练次数、轮数、维度、学习率、客户端数和裁剪阈值必须为正数。")
    if config.pca_dim > config.input_dim:
        parser.error("PCA 维度不能大于输入维度。")
    if 10 % config.num_clients:
        parser.error("当前 non-IID 划分要求客户端数能整除十个数字类别。")
    noise_values = [config.beta, config.ablation_noise_std, *config.noise_multipliers]
    if any(not math.isfinite(value) or value < 0 for value in noise_values):
        parser.error("KL 权重、噪声标准差和噪声倍数必须是有限非负数。")
    if len(set(config.seeds)) != len(config.seeds):
        parser.error("训练种子不能重复，否则会重复统计同一次实验。")
    if any(seed < 0 or seed >= 2**32 for seed in [*config.seeds, config.split_seed, config.partition_seed]):
        parser.error("随机种子必须位于 [0, 2**32) 内。")
    output_dir = args.output_dir
    if output_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output_dir = Path(__file__).resolve().parent / "result" / run_id
    return config, output_dir.resolve(), stage_order(requested)


def main(argv=None):
    config, output_dir, order = parse_args(argv)
    # 命令行运行保存图片，不打开窗口；保持原入口的单线程训练设置。
    import matplotlib
    import torch
    from .artifacts import prepare_run, record_stage, stage_complete

    matplotlib.use("Agg")
    torch.set_num_threads(1)
    try:
        manifest = prepare_run(output_dir, config)
        completed = {stage: stage_complete(output_dir, manifest, stage) for stage in order}
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"Output: {output_dir}\nStages (including dependencies): {order}", flush=True)
    for stage in order:
        if completed[stage]:
            print(f"Stage {stage}: reuse completed results", flush=True)
            continue
        print(f"Stage {stage}: start", flush=True)
        record_stage(output_dir, manifest, stage, "running")
        try:
            stage_dir = output_dir / f"stage{stage}"
            stage_dir.mkdir(exist_ok=True)
            module = import_module(f".experiments.stage{stage}_{STAGES[stage]}", package="week5")
            getattr(module, f"run_stage{stage}")(config, stage_dir)
        except BaseException as error:
            record_stage(output_dir, manifest, stage, "failed", error)
            raise
        record_stage(output_dir, manifest, stage, "complete")
        print(f"Stage {stage}: complete", flush=True)
    print(f"Finished: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
