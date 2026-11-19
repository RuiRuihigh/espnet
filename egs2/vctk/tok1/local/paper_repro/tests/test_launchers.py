"""CPU-only launcher regressions; no installs, scheduler calls, or GPU allocations."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('runpod_launcher', PACKAGE / 'templates/runpod.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def checkpoint(epoch):
    return {'model': {}, 'reporter': {'epoch': epoch},
            'optimizers': [{'state': {1: {}}}, {'state': {2: {}}}],
            'schedulers': [{}, {}]}


class LauncherTests(unittest.TestCase):
    def render(self, backend):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / 'workspace'
        command = ['python3', str(PACKAGE / 'init_workspace.py'), '--backend', backend,
                   '--root', str(root), '--base', '/tmp/assets', '--vctk', '/tmp/vctk',
                   '--libri', '/tmp/libri', '--conda-sh', '/tmp/conda.sh']
        if backend == 'slurm':
            command += ['--cpu-account', 'cpu-acct', '--gpu-account', 'gpu-acct',
                        '--cpu-partition', 'compute', '--gpu-partitions', 'accelerated',
                        '--preflight-partition', 'accelerated']
        subprocess.run(command, check=True, capture_output=True, text=True)
        for path in root.glob('*.py'):
            ast.parse(path.read_text())
        for path in root.glob('*.sh'):
            subprocess.run(['bash', '-n', str(path)], check=True)
        return root, command

    def test_runpod_render_without_scheduler_options(self):
        root, _ = self.render('runpod')
        self.assertTrue((root / 'runpod.py').is_file())
        self.assertFalse((root / 'train_chain.py').exists())
        self.assertFalse((root / 'submit_pipeline.py').exists())
        self.assertFalse(json.loads((root / 'PORTING.json').read_text())['jobs_submitted'])

    def test_nonempty_workspace_is_preserved(self):
        root, command = self.render('runpod')
        before = (root / 'PORTING.json').read_bytes()
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(before, (root / 'PORTING.json').read_bytes())

    def test_symlink_import_check(self):
        root, _ = self.render('runpod')
        tree = ast.parse((root / 'runtime_audit.py').read_text())
        assertion = next(n for n in tree.body if isinstance(n, ast.Assert))
        task = type('Task', (), {'__file__': str(root / 'espnet/espnet2/tasks/gan_tok.py')})
        code = compile(ast.Expression(assertion.test), '<path-check>', 'eval')
        self.assertTrue(eval(code, {'Path': Path, 'task': task, 'R': root}))
        task.__file__ = str(root / 'unrelated/gan_tok.py')
        self.assertFalse(eval(code, {'Path': Path, 'task': task, 'R': root}))

    def test_partition_is_not_torch_device(self):
        root, _ = self.render('slurm')
        tree = ast.parse((root / 'train_chain.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'load']
        self.assertEqual(len(calls), 1)
        value = next(k.value.value for k in calls[0].keywords if k.arg == 'map_location')
        self.assertEqual(value, 'cpu')

    def test_one_and_two_gpu_training_arguments(self):
        for ngpu in [1, 2]:
            cmd = launcher.train_command(Path('/work'), ngpu, 8000000, 31, Path('/work/train'))
            self.assertEqual(cmd[cmd.index('--ngpu') + 1], str(ngpu))
            self.assertEqual(cmd[cmd.index('--num_nodes') + 1], '1')
            self.assertIn('--max_epoch 31 --batch_bins 8000000 --accum_grad 1', cmd)
            self.assertNotIn('sbatch', ' '.join(cmd))
            self.assertIn('/work/python.sh', cmd)

    def test_fresh_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            expected = {'ngpu': 2, 'batch_bins': 16000000}
            self.assertEqual(launcher.check_resume('start', out, expected, None), 0)
            (out / 'checkpoint.pth').touch()
            (out / 'RUNPOD_RUN.json').write_text(json.dumps(expected))
            for epoch in [1, 30, 31, 90]:
                self.assertEqual(launcher.check_resume('resume', out, expected,
                                 lambda _: checkpoint(epoch)), epoch)
            with self.assertRaises(ValueError):
                launcher.check_resume('start', out, expected, None)
            with self.assertRaises(ValueError):
                launcher.check_resume('resume', out, {'ngpu': 1}, None)

    def test_resume_requires_checkpoint_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                launcher.check_resume('resume', Path(d), {}, None)

    def test_reject_partial_training_state(self):
        for key in ['optimizers', 'schedulers']:
            ck = checkpoint(30)
            ck[key] = [{}]
            with self.assertRaises(ValueError):
                launcher.validate_checkpoint(ck)
        ck = checkpoint(30)
        ck['optimizers'][0]['state'] = {}
        with self.assertRaises(ValueError):
            launcher.validate_checkpoint(ck)

    def test_spawned_worker_does_not_restart_training(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            entry = root / 'train_entry.py'
            entry.write_text((PACKAGE / 'templates/train_entry.py').read_text())
            (root / 'torch.py').write_text('def save(*args, **kwargs): pass\n')
            package = root / 'espnet2/bin'
            package.mkdir(parents=True)
            (package.parent / '__init__.py').touch()
            (package / '__init__.py').touch()
            (package / 'gan_tok_train.py').write_text(
                'import multiprocessing\n'
                'def worker():\n'
                '    import torch\n'
                '    assert torch.save.__name__ == "atomic_save"\n'
                'def main():\n'
                '    p = multiprocessing.get_context("spawn").Process(target=worker)\n'
                '    p.start(); p.join(10)\n'
                '    assert p.exitcode == 0, p.exitcode\n')
            subprocess.run([sys.executable, str(entry)], check=True, timeout=20,
                           env={**os.environ, 'PYTHONPATH': str(root)})

    def test_training_flow_uses_last_saved_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for marker in ['SETUP_OK.json', 'DATA_READY.json', 'stages/stats.done']:
                path = root / marker
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            expected = {'ngpu': 2, 'batch_bins': 16000000}
            audit = root / 'preflight_runpod_2gpu_16000000'
            audit.mkdir()
            (audit / 'READY.json').write_text(json.dumps(expected))
            out = root / 'train_runs/runpod'
            out.mkdir(parents=True)
            (out / 'RUNPOD_RUN.json').write_text(json.dumps(expected))
            (out / 'checkpoint.pth').touch()
            # No PROGRESS.json: a completed checkpoint must still resume correctly.
            epochs = iter([30, 31])
            loads = []
            def load(path, **kwargs):
                loads.append(kwargs)
                return checkpoint(next(epochs))
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
                                    device_count=lambda: 2), load=load)
            with patch.object(launcher, 'ROOT', root), \
                 patch.object(launcher, 'signature', return_value=expected), \
                 patch.dict(sys.modules, {'torch': torch}), \
                 patch.object(sys, 'argv', ['runpod.py', 'resume', '--ngpu', '2', '--target-epoch', '31']), \
                 patch.object(launcher.subprocess, 'run') as run:
                launcher.main()
            self.assertEqual(json.loads((out / 'PROGRESS.json').read_text())['epoch'], 31)
            self.assertTrue(all(k['map_location'] == 'cpu' for k in loads))
            self.assertIn('--max_epoch 31 --batch_bins 16000000 --accum_grad 1',
                          run.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
