"""Explicit cpu/cuda device selection; CUDA paths run only where a CUDA device exists."""
import json

import pytest
import torch

from psdl2.checkpoint import save_checkpoint
from psdl2.index import load_encoder
from psdl2.model import Encoder, resolve_device
from psdl2.schema import ContractError
from psdl2.train import main as train_main, train
from test_checkpoint import checkpoint_config
from test_end_to_end import synthetic_exchange
from test_model import model_config

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device not available')


def test_device_names_are_explicit():
    assert resolve_device('cpu') == torch.device('cpu')
    for name in ('gpu', 'mps', 'cuda:0', None):
        with pytest.raises(ContractError, match='device_must_be_cpu_or_cuda'):
            resolve_device(name)


@pytest.mark.skipif(torch.cuda.is_available(), reason='checks the no-CUDA error path')
def test_unavailable_cuda_fails_without_cpu_fallback(tmp_path, capsys):
    with pytest.raises(ContractError, match='cuda_device_unavailable'):
        resolve_device('cuda')
    path, _, _, _, _ = synthetic_exchange(tmp_path)
    code = train_main([str(path), '--config', str(tmp_path/'config.json'), '--out', str(tmp_path/'c.pt'),
                       '--lineages', str(tmp_path/'lineages.json'), '--split', str(tmp_path/'split.json'),
                       '--device', 'cuda'])
    assert code == 2 and 'cuda_device_unavailable' in capsys.readouterr().err
    assert not (tmp_path/'c.pt').exists()


def test_cpu_checkpoint_stores_no_cuda_rng(tmp_path):
    model = Encoder(model_config()); optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    path = tmp_path/'cpu.pt'; save_checkpoint(path, model, optimizer, checkpoint_config(), epoch=1)
    assert torch.load(path, weights_only=True)['rng']['cuda'] == []


def test_cpu_training_reports_device(tmp_path):
    _, export, config, lineages, assignments = synthetic_exchange(tmp_path)
    result = train(export.ready, config, tmp_path/'c.pt', lineages=lineages, assignments=assignments, device='cpu')
    assert result['device'] == 'cpu'


@cuda
def test_cuda_training_checkpoint_loads_on_cpu_and_cuda(tmp_path):
    _, export, config, lineages, assignments = synthetic_exchange(tmp_path)
    path = tmp_path/'c.pt'
    result = train(export.ready, config, path, lineages=lineages, assignments=assignments, device='cuda')
    assert result['device'] == 'cuda'
    assert torch.load(path, map_location='cpu', weights_only=True)['rng']['cuda']
    cpu_model, views, fp_cpu = load_encoder(path, 'cpu')
    gpu_model, _, fp_gpu = load_encoder(path, 'cuda')
    assert fp_cpu == fp_gpu and next(gpu_model.parameters()).is_cuda
    bundle = views.bundle(export.ready[0])
    with torch.no_grad():
        torch.testing.assert_close(gpu_model([bundle])['embedding'].cpu(), cpu_model([bundle])['embedding'],
                                   atol=1e-5, rtol=1e-4)


def test_training_progress_lines_are_json_on_stderr(tmp_path, capsys):
    path, _, _, _, _ = synthetic_exchange(tmp_path)
    args = [str(path), '--config', str(tmp_path/'config.json'), '--out', str(tmp_path/'c.pt'),
            '--lineages', str(tmp_path/'lineages.json'), '--split', str(tmp_path/'split.json')]
    assert train_main(args) == 0
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [(r['phase'], r['state']) for r in lines] == [('train', 'started'), ('train', 'epoch_completed')]
    assert lines[0]['device'] == 'cpu' and lines[1]['optimizer_steps'] == 1
    assert train_main(args[:-4] + ['--lineages', str(tmp_path/'lineages.json'), '--split', str(tmp_path/'split.json'),
                                   '--resume', str(tmp_path/'c.pt'), '--no-progress']) == 2
    assert all('progress' not in line for line in capsys.readouterr().err.splitlines())


def test_running_progress_line_reports_batch_and_loss(tmp_path):
    _, export, config, lineages, assignments = synthetic_exchange(tmp_path)
    lines = []
    train(export.ready, config, tmp_path/'c.pt', lineages=lineages, assignments=assignments,
          progress_sink=lines.append, progress_interval=0)
    running = [r for r in lines if r['state'] == 'running']
    assert running and running[0]['batch'] == 1 and running[0]['batches'] == 1
    assert isinstance(running[0]['loss'], float)
