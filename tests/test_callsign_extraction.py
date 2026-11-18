"""Validate judge-based extraction independently of reference/error scoring."""
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / 'app'))
from judge_client import RemoteCxEREvaluator


def test_callsigns_are_deduplicated_and_use_selected_model():
    response = Mock()
    response.json.return_value = {'choices': [{'message': {'content': '{"callsigns": ["delta 123", "delta 123"]}'}}]}
    with patch('judge_client.requests.post', return_value=response) as post:
        evaluator = RemoteCxEREvaluator('http://localhost:8000/v1', 'qwen')
        assert evaluator.extract_callsigns('delta 123 climb') == ['delta 123']
        assert post.call_args.kwargs['json']['model'] == 'qwen'
        assert post.call_args.args[0].endswith('/chat/completions')


def test_malformed_callsigns_are_not_treated_as_empty_success():
    response = Mock()
    response.json.return_value = {'choices': [{'message': {'content': '{"callsigns": "delta 123"}'}}]}
    with patch('judge_client.requests.post', return_value=response):
        try:
            RemoteCxEREvaluator('http://localhost:8000/v1', 'llama').extract_callsigns('delta 123')
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed extraction must fail')
