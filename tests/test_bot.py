"""Offline tests for bot.py: the built-in self-test (decision rule, validation, hedge, recall wiring) and
the no-retry-on-timeout rule.   python -m unittest discover -s tests -v"""
import importlib.util, os, subprocess, sys, types, unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_bot():
    real = sys.stdout
    spec = importlib.util.spec_from_file_location("bot_under_test", os.path.join(HERE, "bot.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.stdout = real                     # bot.py tees stdout into logs/console.log on import
    return m


class TestSelftest(unittest.TestCase):
    def test_builtin_selftest_passes(self):
        env = dict(os.environ, SELFTEST="1", LOCK_FILE="unittest.lock")
        p = subprocess.run([sys.executable, os.path.join(HERE, "bot.py")], cwd=HERE, env=env,
                           capture_output=True, text=True, timeout=120)
        self.assertIn("selftest: 0 failure(s)", p.stdout + p.stderr, (p.stdout + p.stderr)[-2000:])


class TestNoRetryOnTimeout(unittest.TestCase):
    def test_timeout_is_not_retried_but_format_errors_are(self):
        bot = load_bot()
        calls = []

        class APITimeoutError(Exception):
            pass

        class FakeCompletions:
            def __init__(self, exc):
                self.exc = exc

            def create(self, **kw):
                calls.append(kw)
                if self.exc is not None and "response_format" in kw:
                    raise self.exc
                msg = types.SimpleNamespace(content='{"items": [{"index": 1, "median": 10}]}')
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None)

        def fake_openai(exc):
            def factory(**kw):
                return types.SimpleNamespace(chat=types.SimpleNamespace(completions=FakeCompletions(exc)))
            return factory

        import openai
        real = openai.OpenAI
        try:
            openai.OpenAI = fake_openai(APITimeoutError("Request timed out."))
            with self.assertRaises(APITimeoutError):
                bot.llm_items("p", "d", "i", 5.0)
            self.assertEqual(len(calls), 1, "a timed-out call must not be sent again")
            calls.clear()
            openai.OpenAI = fake_openai(ValueError("response_format not supported"))
            items, _ = bot.llm_items("p", "d", "i", 5.0)
            self.assertEqual(items, [{"index": 1, "median": 10}])
            self.assertEqual(len(calls), 2, "a format error gets exactly one plain retry")
        finally:
            openai.OpenAI = real


if __name__ == "__main__":
    unittest.main(verbosity=2)
