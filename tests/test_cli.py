#!/usr/bin/env python3
"""Bounded CLI/C ABI transport checks; optional --real uses accepted GGUF fixtures."""
import argparse
import csv
import ctypes as C
import json
import math
import os
from pathlib import Path
import struct
import subprocess


class Request(C.Structure):
    _fields_ = [("context", C.POINTER(C.c_float)), ("mask", C.POINTER(C.c_uint8)),
                ("B", C.c_int32), ("V", C.c_int32), ("T", C.c_int32), ("H", C.c_int32),
                ("past", C.POINTER(C.c_float)), ("P", C.c_int32),
                ("known", C.POINTER(C.c_float)), ("K", C.c_int32),
                ("cp", C.POINTER(C.c_double)), ("CP", C.c_int32),
                ("ck", C.POINTER(C.c_double)), ("CK", C.c_int32),
                ("symmetric", C.c_int32), ("clip", C.c_int32), ("cap", C.c_int32)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cli", required=True); p.add_argument("--library", required=True)
    p.add_argument("--model", required=True); p.add_argument("--scratch", required=True)
    p.add_argument("--real", action="store_true")
    a = p.parse_args(); root = Path(a.scratch); root.mkdir(parents=True, exist_ok=True)
    lib = C.CDLL(a.library)
    lib.timestral_capi_load.argtypes = [C.c_char_p, C.c_int]; lib.timestral_capi_load.restype = C.c_void_p
    lib.timestral_capi_last_error.argtypes = [C.c_void_p]; lib.timestral_capi_last_error.restype = C.c_char_p
    lib.timestral_capi_forecast.argtypes = [C.c_void_p, C.POINTER(Request), C.POINTER(C.c_int32)]
    lib.timestral_capi_forecast.restype = C.POINTER(C.c_float)
    lib.timestral_capi_free.argtypes = [C.c_void_p]; lib.timestral_capi_free_buffer.argtypes = [C.c_void_p]
    checks = comparisons = 0

    def run(args, code=0):
        nonlocal checks
        r = subprocess.run([a.cli] + args, capture_output=True, text=True, timeout=240)
        assert r.returncode == code, (args, r.returncode, r.stderr)
        if code: assert not r.stdout and r.stderr
        checks += 1
        return r

    def packed_bytes(values):
        return struct.pack("=" + "f" * len(values), *values)

    def compare(name, arrays, B, V, T, H, explicit=False, csv_output=False):
        nonlocal comparisons
        cmd = ["forecast", "--model", a.model, "--horizon", str(H)]
        keep = []
        req = Request(); req.B, req.V, req.T, req.H = B, V, T, H
        req.symmetric = req.clip = 0 if explicit else -1
        for key, field, count, width, rows, ctype in [
                ("context", "context", None, V, T, C.c_float),
                ("past", "past", "P", arrays.get("P", 0), T, C.c_float),
                ("known", "known", "K", arrays.get("K", 0), T + H, C.c_float),
                ("cp", "cp", "CP", arrays.get("CP", 0), T, C.c_double),
                ("ck", "ck", "CK", arrays.get("CK", 0), T + H, C.c_double)]:
            if not width: continue
            values = arrays[key]; buf = (ctype * len(values))(*values); keep.append(buf); setattr(req, field, buf)
            if count: setattr(req, count, width)
            flag = {"context": "--context", "past": "--past-cov", "known": "--known-cov", "cp": "--cat-past", "ck": "--cat-known"}[key]
            for b in range(B):
                path = root / f"{name}-{key}-{b}.csv"
                with path.open("w", newline="") as f:
                    writer = csv.writer(f)
                    for t in range(rows):
                        row = [values[(b * width + v) * rows + t] for v in range(width)]
                        writer.writerow([("NaN" if t % 2 else "") if math.isnan(float(x)) else repr(float(x)) for x in row])
                cmd += [flag, str(path)]
        if explicit: cmd += ["--no-symmetric", "--no-clip", "--threads", "2"]
        handle = lib.timestral_capi_load(os.fsencode(a.model), 2 if explicit else 1)
        assert handle, lib.timestral_capi_last_error(None)
        try:
            Q = C.c_int32(); out = lib.timestral_capi_forecast(handle, C.byref(req), C.byref(Q))
            assert out, lib.timestral_capi_last_error(handle)
            try: expected = C.string_at(out, B * V * H * Q.value * 4)
            finally: lib.timestral_capi_free_buffer(out)
        finally: lib.timestral_capi_free(handle)
        result = json.loads(run(cmd + ["--json"]).stdout)
        assert result["shape"] == [B, V, H, Q.value]
        assert packed_bytes(result["values"]) == expected
        comparisons += 1
        if csv_output:
            rows = list(csv.reader(run(cmd).stdout.splitlines()))
            assert rows[0][:3] == ["series", "variate", "step"]
            assert [float(x[2:]) for x in rows[0][3:]] == result["quantiles"]
            values = []
            for i, row in enumerate(rows[1:]):
                assert list(map(int, row[:3])) == [i // (V * H), i // H % V, i % H]
                values.extend(map(float, row[3:]))
            assert packed_bytes(values) == expected
            comparisons += 1
        print("TRANSPORT", name, "bytes", len(expected), "PASS")

    if a.real:
        import gguf
        import numpy as np
        suite = os.environ.get("TIMESTRAL_TEST_REAL_BASELINES")
        if not suite: return 77
        for name, H in [("uni_short", 96), ("known_cov", 64)]:
            reader = gguf.GGUFReader(str(Path(suite) / "real" / f"baseline-{name}-f32.gguf"))
            tensors = {t.name: t for t in reader.tensors}
            t = tensors["input.context"]; dims = list(t.shape) + [1] * 4
            arrays = {"context": np.asarray(t.data).reshape(-1).tolist()}
            for key, source, count in [("past", "input.past", "P"), ("known", "input.known", "K")]:
                if source in tensors:
                    arrays[key] = np.asarray(tensors[source].data).reshape(-1).tolist()
                    arrays[count] = int(tensors[source].shape[1])
            compare(name, arrays, int(dims[2]), int(dims[1]), int(dims[0]), H, csv_output=True)
    else:
        info = json.loads(run(["info", "--model", a.model, "--json"]).stdout)
        assert info["patch_size"] > 0 and info["quantiles"] and "values" not in info
        assert "patch_size:" in run(["info", "--model", a.model]).stdout
        run(["quantize"], 2); run(["forecast"], 2); run(["info", "--model", "absent"], 1)
        basic = {"context": [float(i % 7 - 3) for i in range(32)]}
        compare("basic", basic, 1, 1, 32, 8, csv_output=True)
        missing = {"context": [math.nan if i % 11 == 0 else float(i % 9 - 4) for i in range(128)]}
        compare("batch-missing", missing, 2, 2, 32, 8, explicit=True, csv_output=True)
        cov = {"context": [float(i % 7) for i in range(64)], "P": 1, "K": 1, "CP": 1, "CK": 1,
               "past": [float(i % 5) for i in range(64)], "known": [float(i % 3) for i in range(80)],
               "cp": [float(i % 3) for i in range(64)], "ck": [float(i % 2) for i in range(80)]}
        compare("batch-all-cov", cov, 2, 1, 32, 8, csv_output=True)
        good = root / "good.csv"; good.write_text("1\n2\n")
        base = ["forecast", "--model", a.model, "--context", str(good), "--horizon", "1"]
        for token in ["inf", "-inf", "1junk", "header", "0x10", "nan(payload)", "1e999", "1e100"]:
            bad = root / "bad.csv"; bad.write_text(token + "\n")
            r = run(["forecast", "--model", a.model, "--context", str(bad), "--horizon", "1"], 2)
            assert str(bad) + ":1:1:" in r.stderr
        for args in [["--horizon", "0"], ["--threads", "0"], ["--threads", "1025"], ["--oops"], ["--json", "--json"]]: run(base + args, 2)
        ragged = root / "ragged.csv"; ragged.write_text("1,2\n3\n")
        run(["forecast", "--model", a.model, "--context", str(ragged), "--horizon", "1"], 2)
        run(base + ["--context", str(good), "--past-cov", str(good)], 2)
        short = root / "short.csv"; short.write_text("1\n")
        wide = root / "wide.csv"; wide.write_text("1,2\n3,4\n")
        run(base + ["--context", str(short)], 2)
        run(base + ["--context", str(wide)], 2)
        run(base + ["--known-cov", str(good)], 2)
        run(base + ["--context", str(good), "--past-cov", str(good), "--past-cov", str(wide)], 2)
        upper = root / "upper.csv"; upper.write_text("+1.0e0\r\n nAn \r\n")
        run(["forecast", "--model", a.model, "--context", str(upper), "--horizon", "1", "--json"])
        run(base[:-1] + ["1", "--model", "absent"], 2)  # duplicate singleton
        missing_model = list(base); missing_model[2] = "absent"; run(missing_model, 1)

        bench = json.loads(run(["bench", "--model", a.model, "--context-len", "32", "--horizon", "8", "--threads", "1", "--repeat", "2", "--warmup", "1"]).stdout)
        assert bench["input"] == "dyadic-v1" and bench["input_sum"] == -.515625 and len(bench["samples_seconds"]) == 2
        assert bench["repeat"] == 2 and bench["warmup"] == 1 and bench["output_sum"] == bench["output_sum"]
    print(f"PASS cli process_checks={checks} bitwise_comparisons={comparisons} real={a.real}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
