from dataclasses import asdict
import importlib
import json
from pathlib import Path
import runpy

import pytest
import torch

from psdl2.checkpoint import save_checkpoint
from psdl2.data import load_export
from psdl2.index import candidate_record
from psdl2.model import Encoder
from psdl2.schema import ContractError
from psdl2.train import read_config
from test_model import model_config
from test_views import view_config


@pytest.mark.parametrize('module', ['train', 'index', 'search', 'evaluate'])
def test_module_help_with_sockets_blocked(module, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, 'argv', ['psdl2.'+module, '--help'])
    with pytest.raises(SystemExit) as status:
        runpy.run_module('psdl2.'+module, run_name='__main__')
    assert status.value.code == 0 and 'offline' in capsys.readouterr().out


def test_config_no_labels_and_requires_geometry():
    path = Path(__file__).parents[1]/'configs/synthetic.example.json'
    config = read_config(path)
    assert 'labels' not in config and 'jitter_xy_um' in config['views']


def test_index_search_evaluate_cli_nontraining_roundtrip(source, tmp_path, capsys):
    from psdl2.index import main as index_main
    from psdl2.search import main as search_main
    from psdl2.evaluate import main as evaluate_main
    _, write, root = source; export_path = write(); roi = load_export(export_path).ready[0]
    config = read_config(Path(__file__).parents[1]/'configs/synthetic.example.json')
    model = Encoder(model_config()); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    checkpoint = tmp_path/'weights.pt'; save_checkpoint(checkpoint, model, optimizer, config, epoch=0)
    index_path = tmp_path/'index.json'
    assert index_main([str(export_path), '--checkpoint', str(checkpoint), '--out', str(index_path)]) == 0
    assert json.loads(capsys.readouterr().out)['n_indexed'] == 1
    # Fixed synthetic candidate sharing the compatible case but belonging to cohort A.
    index = json.loads(index_path.read_text()); candidate = index['records'][0]
    candidate['roi_id'] = 'candidate_A'; candidate['product_id'] = 'A'; candidate['source_identity']['product_id'] = 'A'
    index_path.write_text(json.dumps(index)); cohort = tmp_path/'cohort.json'; cohort.write_text('["A"]')
    result_path = tmp_path/'result.json'
    args = [str(export_path), '--roi-id', roi.row['roi_id'], '--checkpoint', str(checkpoint),
            '--index', str(index_path), '--cohort', str(cohort), '--topk', '1', '--out', str(result_path)]
    assert search_main(args) == 0
    capsys.readouterr(); result = json.loads(result_path.read_text())
    assert result['primary'][0]['roi_id'] == 'candidate_A'
    assert result['primary'][0]['score'] == 1
    results = tmp_path/'results.json'; results.write_text(json.dumps([result]))
    split = tmp_path/'evaluation_split.json'; split.write_text(json.dumps({roi.row['roi_id']: 'test'}))
    eval_args = [str(results), '--k', '1', '--split', str(split), '--partition', 'test']
    assert evaluate_main(eval_args) == 3
    assert json.loads(capsys.readouterr().out)['recall_at_k'] is None
    labels = tmp_path/'labels.json'; labels.write_text(json.dumps({'schema_version': 'roi_relations_v1',
        'relations': [{'a': roi.row['roi_id'], 'b': 'candidate_A', 'relation': 'similar'}]}))
    assert evaluate_main(eval_args+['--labels', str(labels)]) == 0
    assert json.loads(capsys.readouterr().out)['recall_at_k'] == 1
    index['view_fingerprint'] = 'c'*64; index_path.write_text(json.dumps(index))
    assert search_main(args) == 2
    assert 'incompatible_index' in capsys.readouterr().err
