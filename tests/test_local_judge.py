"""Judge launch configuration and ownership without downloading models."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('local_judge', Path(__file__).parents[1] / 'app/local_judge.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_launch_preserves_parent_gpu_visibility_and_uses_selected_devices(tmp_path):
    manager = module.LocalJudge(tmp_path / 'judge.log')
    process = Mock()
    process.poll.return_value = None
    with patch.dict(module.os.environ, {'CUDA_VISIBLE_DEVICES': '1,2,3,4'}):
        with patch.object(module.subprocess, 'run', return_value=Mock(returncode=0)), patch.object(module.subprocess, 'Popen', return_value=process) as launch:
            manager.start(module.JUDGE_MODELS['Qwen 2.5 72B'], ['2', '3'], 8001, 0.8)
            args, kwargs = launch.call_args
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '2,3'
            assert module.os.environ['CUDA_VISIBLE_DEVICES'] == '1,2,3,4'
            assert args[0][args[0].index('--tensor-parallel-size') + 1] == '2'
            assert kwargs['start_new_session']
            assert args[0][0] == str(Path(module.__file__).resolve().parents[1] / '.venv-judge/bin/python')
            assert args[0][1] == '-I'
            assert 'PYTHONPATH' not in kwargs['env']
            assert kwargs['env']['PYTHONNOUSERSITE'] == '1'
            try:
                manager.start('another', ['4'], 8001, 0.8)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Running judge must not be replaced')
            launch.assert_called_once()


def test_stop_targets_owned_process_group(tmp_path):
    manager = module.LocalJudge(tmp_path / 'judge.log')
    manager.process = Mock(pid=12345)
    manager.process.poll.return_value = None
    with patch.object(module.os, 'killpg') as kill:
        manager.stop()
        kill.assert_called_once_with(12345, module.signal.SIGTERM)
    manager.process.wait.assert_called_once_with(timeout=10)


def test_missing_vllm_fails_before_starting_process(tmp_path):
    manager = module.LocalJudge(tmp_path / 'judge.log')
    with patch.object(module.subprocess, 'run', return_value=Mock(returncode=1, stderr='')):
        with patch.object(module.subprocess, 'Popen') as launch:
            try:
                manager.start('judge', ['1'], 8000, 0.8)
            except RuntimeError as error:
                assert 'scripts/setup_local.sh judge' in str(error)
            else:
                raise AssertionError('Missing vLLM should fail')
            launch.assert_not_called()
