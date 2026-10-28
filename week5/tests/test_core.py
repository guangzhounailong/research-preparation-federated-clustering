"""保护重排涉及的参数传递、聚合、隐私与结果复用行为。"""

from contextlib import redirect_stderr
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from week5.artifacts import prepare_run, record_stage, stage_complete
from week5.client import train_client
from week5.config import DEFAULT_CONFIG
from week5.main import parse_args, stage_order
from week5.models import Autoencoder, VAE
from week5.privacy import clip_client_update
from week5.server import fedavg, fedavg_with_noise
from week5.training import extract_features, train_model


class TrainingTests(unittest.TestCase):
    def test_fedavg_uses_sample_weights(self):
        states = [{"w": torch.tensor([1.0, 3.0])}, {"w": torch.tensor([5.0, 7.0])}]
        result = fedavg(states, [1, 3])
        torch.testing.assert_close(result["w"], torch.tensor([4.0, 6.0]))

    def test_clipping_uses_joint_update_norm(self):
        global_state = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        local_state = {"a": torch.tensor([4.0]), "b": torch.tensor([6.0])}
        update, before, after = clip_client_update(local_state, global_state, 2.0)
        self.assertEqual(before, 5.0)
        self.assertAlmostEqual(after, 2.0)
        torch.testing.assert_close(update["a"], torch.tensor([1.2]))
        torch.testing.assert_close(update["b"], torch.tensor([1.6]))

    def test_fixed_noise_std_works_without_clipping(self):
        states = [{"w": torch.tensor([3.0, 5.0])}, {"w": torch.tensor([7.0, 9.0])}]
        global_state = {"w": torch.tensor([1.0, 1.0])}
        result, norms = fedavg_with_noise(
            states, [1, 3], global_state, None, 0.0,
            torch.Generator().manual_seed(47), noise_std=0.005,
        )
        expected = torch.tensor([6.0, 8.0]) + torch.randn(
            (2,), generator=torch.Generator().manual_seed(47),
        ) * 0.005
        torch.testing.assert_close(result["w"], expected, rtol=0, atol=0)
        for record in norms:
            self.assertEqual(record["Norm_before"], record["Norm_after"])

    def test_client_honors_config_without_changing_global_model(self):
        config = replace(DEFAULT_CONFIG, input_dim=4, hidden_dim=6, latent_dim=2,
                         local_epochs=1, epochs=3, batch_size=4, learning_rate=0.01)
        torch.manual_seed(47)
        model = Autoencoder(config=config)
        original = deepcopy(model.state_dict())
        expected = deepcopy(model)
        data = np.arange(32, dtype=np.float32).reshape(8, 4) / 32
        state, size = train_client(model, data, 47, config=config)
        train_model(expected, data, 47, epochs=1, config=config)
        self.assertEqual(size, 8)
        self.assertTrue(any(not torch.equal(state[name], original[name]) for name in state))
        for name in state:
            torch.testing.assert_close(state[name], expected.state_dict()[name], rtol=0, atol=0)
            torch.testing.assert_close(model.state_dict()[name], original[name], rtol=0, atol=0)

    def test_vae_features_are_posterior_means(self):
        config = replace(DEFAULT_CONFIG, input_dim=4, hidden_dim=6, latent_dim=2)
        model = VAE(config=config)
        data = np.ones((3, 4), dtype=np.float32)
        expected, _ = model.distribution(torch.from_numpy(data))
        np.testing.assert_array_equal(extract_features(model, data), expected.detach().numpy())

    def test_ae_features_are_l2_normalized_with_zero_vectors_preserved(self):
        config = replace(DEFAULT_CONFIG, input_dim=2, hidden_dim=2, latent_dim=2)
        model = Autoencoder(config=config)
        with torch.no_grad():
            model.encoder[0].weight.copy_(torch.eye(2))
            model.encoder[0].bias.zero_()
            model.encoder[2].weight.copy_(torch.diag(torch.tensor([3.0, 4.0])))
            model.encoder[2].bias.zero_()
        data = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        features = extract_features(model, data)
        np.testing.assert_allclose(features, [[0.6, 0.8], [0.0, 0.0]], atol=1e-7)
        # 编码器本身仍输出原始特征，训练和重建不会使用归一化后的表示。
        np.testing.assert_array_equal(
            model.encode(torch.from_numpy(data)).detach().numpy(),
            [[3.0, 4.0], [0.0, 0.0]],
        )


class EntryPointTests(unittest.TestCase):
    def test_overrides_do_not_mutate_defaults(self):
        config, _, order = parse_args(["--stage", "6", "--rounds", "2", "--seeds", "48"])
        self.assertEqual(config.communication_rounds, 2)
        self.assertEqual(config.seeds, (48,))
        self.assertEqual(order, [6])
        self.assertEqual(DEFAULT_CONFIG.communication_rounds, 200)
        with self.assertRaises(FrozenInstanceError):
            config.epochs = 1

    def test_dependencies_are_ordered_and_unique(self):
        self.assertEqual(stage_order([8]), [1, 3, 4, 7, 8])
        self.assertEqual(stage_order([9]), [1, 3, 4, 5, 7, 9])
        self.assertEqual(stage_order(list(range(1, 10))), list(range(1, 10)))

    def test_invalid_arguments_fail_before_training(self):
        examples = [
            ["--stage", "10"], ["--rounds", "0"], ["--seeds", "47", "47"],
            ["--noise-multipliers", "nan"], ["--num-clients", "3"],
        ]
        for arguments in examples:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(arguments)


class ArtifactTests(unittest.TestCase):
    def test_config_mismatch_is_rejected(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)
            prepare_run(path, DEFAULT_CONFIG)
            with self.assertRaisesRegex(ValueError, "配置不同"):
                prepare_run(path, replace(DEFAULT_CONFIG, communication_rounds=2))

    def test_incomplete_stage_is_resumable_and_completed_files_are_checked(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)
            manifest = prepare_run(path, DEFAULT_CONFIG)
            record_stage(path, manifest, 1, "running")
            self.assertFalse(stage_complete(path, manifest, 1))
            stage_dir = path / "stage1"
            stage_dir.mkdir()
            (stage_dir / "stage1_results.csv").write_text("ARI\n0.5000\n")
            record_stage(path, manifest, 1, "complete")
            resumed = prepare_run(path, DEFAULT_CONFIG)
            self.assertTrue(stage_complete(path, resumed, 1))
            (stage_dir / "stage1_results.csv").write_text("ARI\n0.9000\n")
            with self.assertRaisesRegex(ValueError, "缺失或已修改"):
                stage_complete(path, resumed, 1)

    def test_unregistered_results_are_rejected(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "old.csv").write_text("ARI\n0.5\n")
            with self.assertRaisesRegex(ValueError, "未登记"):
                prepare_run(path, DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
