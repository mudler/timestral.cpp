"""Pure M7 metric primitives. No model loading, anchor extraction, or acceptance runner."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Score:
    value: float | None
    status: str = 'ok'

    def json(self):
        """Strict JSON-safe tagged values, never nonstandard NaN/Infinity literals."""
        if self.value is None:
            return dict(value=None, kind='undefined', status=self.status)
        if math.isinf(self.value):
            return dict(value=None, kind='positive-infinity' if self.value > 0 else
                        'negative-infinity', status=self.status)
        if not math.isfinite(self.value):
            raise ValueError('NaN score')
        return dict(value=float(self.value), kind='finite', status=self.status)


def finite(a):
    a = np.asarray(a, dtype=np.float64)
    if not a.size or not np.all(np.isfinite(a)):
        raise ValueError('nonempty finite values required')
    return a


def normalized(f32, quantized, loc, scale):
    """One target row (H,Q), with caller-supplied frozen native F32 anchors."""
    f, q = finite(f32), finite(quantized)
    if f.shape != q.shape or f.ndim != 2:
        raise ValueError('matching (H,Q) required')
    if not math.isfinite(loc) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('invalid frozen anchor')
    z = np.arcsinh((f-loc)/scale)
    w = scale*np.cosh(z)
    rms = float(np.sqrt(np.mean(z*z)))
    if not np.all(np.isfinite(w)) or np.any(w <= 0) or not math.isfinite(rms):
        raise ValueError('invalid normalization frame')
    if rms == 0:
        same = np.array_equal(f, q)
        s = Score(0. if same else math.inf,
                  'zero-rms-identical' if same else 'zero-rms-different')
        return dict(mean=s, maximum=s)
    e = np.abs(q-f)/(w*rms)
    return dict(mean=Score(float(np.mean(e))), maximum=Score(float(np.max(e))))


def wql(truth, prediction, levels):
    y, p, tau = finite(truth), finite(prediction), finite(levels)
    if y.ndim != 1 or tau.ndim != 1 or p.shape != (y.size, tau.size):
        raise ValueError('expected truth (H), predictions (H,Q), levels (Q)')
    if np.any(tau <= 0) or np.any(tau >= 1) or np.any(np.diff(tau) <= 0):
        raise ValueError('strictly increasing quantile levels in (0,1) required')
    u = y[:, None]-p
    n = float(2/tau.size*np.sum(u*(tau-(u < 0))))
    d = float(np.sum(np.abs(y)))
    if d == 0:
        return Score(0. if n == 0 else math.inf,
                     'zero-truth-exact' if n == 0 else 'zero-truth-inexact')
    return Score(n/d)


def mase(truth, median, context, max_context=32768):
    y, p = finite(truth), finite(median)
    x = np.asarray(context, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or x.ndim != 1 or not x.size or max_context <= 0:
        raise ValueError('invalid MASE dimensions')
    x = x[-max_context:]
    observed = np.isfinite(x)
    pairs = observed[1:] & observed[:-1]
    if not np.any(pairs):
        return Score(None, 'no-observed-adjacent-pairs')
    d = float(np.mean(np.abs(x[1:][pairs]-x[:-1][pairs])))
    a = float(np.mean(np.abs(y-p)))
    if d == 0:
        return Score(0. if a == 0 else math.inf,
                     'zero-scale-exact' if a == 0 else 'zero-scale-inexact')
    return Score(a/d)


def average(scores):
    if not scores:
        raise ValueError('empty aggregate')
    if any(s.value is None for s in scores):
        return Score(None, 'undefined-member')
    if any(math.isnan(s.value) or s.value < 0 for s in scores):
        raise ValueError('invalid nonnegative metric')
    if any(math.isinf(s.value) for s in scores):
        return Score(math.inf, 'infinite-member')
    return Score(math.fsum(s.value for s in scores)/len(scores))


def summarize(items):
    """items: {item_name: [row scores]}; equal rows, then equal items, no exclusions."""
    if not items:
        raise ValueError('empty items')
    means = {name: average(rows) for name, rows in items.items()}
    statuses = {}
    for rows in items.values():
        for s in rows:
            statuses[s.status] = statuses.get(s.status, 0)+1
    return dict(aggregate=average(list(means.values())).json(),
                items={k: s.json() for k, s in means.items()},
                item_count=len(items), row_count=sum(map(len, items.values())),
                row_status_counts=statuses)


def relative_change(f32, quantized):
    f, q = f32.value, quantized.value
    if f is None or q is None:
        s = Score(None, 'undefined-loss')
    elif f < 0 or q < 0 or math.isnan(f) or math.isnan(q):
        raise ValueError('invalid loss')
    elif math.isinf(f):
        s = Score(None, 'indeterminate-infinite-baseline')
    elif f == 0:
        s = Score(0. if q == 0 else math.inf,
                  'zero-baseline-exact' if q == 0 else 'zero-baseline-degraded')
    else:
        s = Score(100*(q-f)/f)
    return dict(signed=s, absolute=Score(None if s.value is None else abs(s.value), s.status))


def coverage(expected_names, outcomes):
    """Error matches remain nonnumeric coverage gaps, never synthetic zero scores."""
    if len(set(expected_names)) != len(expected_names) or set(expected_names) != set(outcomes):
        raise ValueError('coverage identities differ')
    allowed = {'forecast', 'expected-error-match', 'error-mismatch', 'missing'}
    if not set(outcomes.values()) <= allowed:
        raise ValueError('unknown coverage status')
    gaps = {k: v for k, v in outcomes.items() if v != 'forecast'}
    return dict(numeric_acceptance='HOLD' if gaps else 'READY',
                required=len(expected_names), forecasts=len(outcomes)-len(gaps), gaps=gaps)


def monotonic(means):
    """Strict F32/F16/Q8_0/Q6_K/Q5_K/Q4_K normalized-mean order only."""
    if len(means) != 6:
        raise ValueError('six ordered formats required')
    values = [s.value for s in means]
    return (all(v is not None and math.isfinite(v) for v in values)
            and values[0] == 0 and all(a < b for a, b in zip(values, values[1:])))
