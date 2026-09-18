import os
import time


class DeepSeekProvider:
    def __init__(self):
        self.model = os.environ.get(
            'DEEPSEEK_MODEL',
            'deepseek-v4-flash'
        )
        self.client = None

    def _build_client(self):
        if self.client is not None:
            return self.client

        api_key = os.environ.get('DEEPSEEK_API_KEY')

        if not api_key:
            raise RuntimeError(
                'DEEPSEEK_API_KEY is not configured'
            )

        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url='https://api.deepseek.com',
            timeout=45.0,
            max_retries=0
        )

        return self.client


    def generate(self, prompt):
        started = time.perf_counter()

        try:
            client = self._build_client()
            from openai import (
                APITimeoutError,
                APIConnectionError,
                APIStatusError,
            )
            response = client.responses.create(
                model=self.model,
                input=prompt,
                reasoning={'effort': 'low'},
                max_output_tokens=512
            )
        except APITimeoutError as exc:
            return {
                'status': 'API_TIMEOUT',
                'error': str(exc),
                'latency_ms': (time.perf_counter() - started) * 1000.0,
                'text': ''
            }
        except APIConnectionError as exc:
            return {
                'status': 'API_CONNECTION_ERROR',
                'error': str(exc),
                'latency_ms': (time.perf_counter() - started) * 1000.0,
                'text': ''
            }
        except APIStatusError as exc:
            return {
                'status': 'API_STATUS_ERROR',
                'error': str(exc),
                'latency_ms': (time.perf_counter() - started) * 1000.0,
                'text': ''
            }
        except Exception as exc:
            return {
                'status': 'API_ERROR',
                'error': str(exc),
                'latency_ms': (time.perf_counter() - started) * 1000.0,
                'text': ''
            }

        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = response.usage

        details = getattr(
            usage,
            'output_tokens_details',
            None
        )

        reasoning_tokens = 0

        if details is not None:
            reasoning_tokens = getattr(
                details,
                'reasoning_tokens',
                0
            ) or 0

        return {
            'status': 'MODEL_OK',
            'response_id': getattr(response, 'id', None),
            'model': self.model,
            'text': getattr(response, 'output_text', '') or '',
            'response_status': getattr(response, 'status', None),
            'input_tokens': getattr(usage, 'input_tokens', 0) or 0,
            'output_tokens': getattr(usage, 'output_tokens', 0) or 0,
            'reasoning_tokens': reasoning_tokens,
            'total_tokens': getattr(usage, 'total_tokens', 0) or 0,
            'latency_ms': latency_ms
        }