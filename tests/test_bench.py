from gkl.bench import measure


def fake_clock(ms):
    calls = []

    def clock(fn):
        calls.append(ms)
        return ms

    return clock, calls


def test_slow_kernel_gets_minimum_reps_not_budget_overrun():
    clock, calls = fake_clock(800.0)  # naive matmul at 4096 territory
    t = measure(lambda: None, clock, warmup=1, budget_ms=1000, min_reps=5)
    assert t["reps"] == 5 and len(calls) == 6


def test_fast_kernel_is_capped():
    clock, _ = fake_clock(0.01)
    assert measure(lambda: None, clock, budget_ms=1000, max_reps=100)["reps"] == 100


def test_median_ignores_one_outlier():
    times = iter([9.0, 1.0, 50.0, 1.0, 1.0, 1.0, 1.0])  # warmup, then 6 timed runs with one outlier

    t = measure(lambda: None, lambda fn: next(times), warmup=1, budget_ms=5, min_reps=1)
    assert t["median_ms"] == 1.0
