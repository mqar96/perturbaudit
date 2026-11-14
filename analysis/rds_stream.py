"""Streaming reader for R .rds files (XDR serialization v2/v3) that writes large numeric vectors straight to disk.

Memory stays small regardless of file size: vectors longer than BIG elements are streamed in chunks to
<outdir>/vec_<n>.bin (raw little-endian int32/float64/uint8) and replaced by a BigVec handle.
Supports gzip/xz/bzip2 compression, S4 objects, ALTREP compact sequences and wrapped vectors, environments,
pairlists, references. Returns a Python tree: dict (named lists, attributes, S4 slots), list, numpy arrays,
BigVec handles, strings.
Usage (CLI): rds_stream.py file.rds outdir   -> prints the tree outline and writes outdir/tree.pkl"""
import sys, os, gzip, lzma, bz2, struct, pickle
import numpy as np

BIG = 2_000_000
CHUNK = 1 << 22

class BigVec:
    def __init__(self, path, dtype, n):
        self.path, self.dtype, self.n = path, dtype, n
    def load(self, mmap=True):
        return np.memmap(self.path, dtype=self.dtype, mode="r", shape=(self.n,)) if mmap else np.fromfile(self.path, dtype=self.dtype, count=self.n)
    def __repr__(self):
        return f"BigVec({self.dtype}, n={self.n:,}, {os.path.basename(self.path)})"

class Sym(str):
    pass

class Env(dict):
    pass

def _open(path):
    with open(path, "rb") as f:
        m = f.read(6)
    if m[:2] == b"\x1f\x8b": return gzip.open(path, "rb")
    if m[:6] == b"\xfd7zXZ\x00": return lzma.open(path, "rb")
    if m[:3] == b"BZh": return bz2.open(path, "rb")
    return open(path, "rb")

class Reader:
    def __init__(self, path, outdir):
        self.f = _open(path); self.out = outdir; os.makedirs(outdir, exist_ok=True)
        self.refs = []; self.nbig = 0; self.bytes_big = 0

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n: raise EOFError
        return b
    def i32(self): return struct.unpack(">i", self.read(4))[0]
    def f64(self): return struct.unpack(">d", self.read(8))[0]
    def length(self):
        n = self.i32()
        if n == -1:
            hi, lo = self.i32(), self.i32(); n = (hi << 32) + lo
        return n
    def string(self, n): return self.read(n).decode("utf-8", errors="replace")

    def header(self):
        fmt = self.read(2)
        if fmt != b"X\n": raise ValueError(f"only XDR format supported, got {fmt!r}")
        ver = self.i32(); self.i32(); self.i32()
        if ver == 3:
            ne = self.i32(); self.read(ne)
        return ver

    def numvec(self, n, code, width, outdtype):
        if n <= BIG:
            return np.frombuffer(self.read(n * width), dtype=code).astype(outdtype)
        p = os.path.join(self.out, f"vec_{self.nbig:04d}.bin"); self.nbig += 1
        with open(p, "wb") as g:
            left = n
            while left:
                k = min(left, CHUNK)
                np.frombuffer(self.read(k * width), dtype=code).astype(outdtype).tofile(g); left -= k
        self.bytes_big += n * np.dtype(outdtype).itemsize
        return BigVec(p, outdtype, n)

    def item(self):
        flags = self.i32()
        t = flags & 0xFF; has_attr = bool(flags & (1 << 9)); has_tag = bool(flags & (1 << 10))
        if t == 254: return None                      # NILVALUE
        if t in (253, 242, 241, 250): return Env()   # GLOBALENV, EMPTYENV, BASEENV, BASENAMESPACE
        if t in (252, 251): return None               # UNBOUNDVALUE, MISSINGARG
        if t == 255:                                  # REFSXP
            idx = flags >> 8
            if idx == 0: idx = self.i32()
            return self.refs[idx - 1]
        if t in (249, 248, 247):                      # NAMESPACE, PACKAGE, PERSIST
            self.i32(); n = self.i32(); v = [self.item() for _ in range(n)]
            self.refs.append(v); return v
        if t == 1:                                    # SYMSXP
            s = Sym(self.item() or ""); self.refs.append(s); return s
        if t == 4:                                    # ENVSXP
            e = Env(); self.refs.append(e); self.i32()
            enclos = self.item(); frame = self.item(); hashtab = self.item(); attr = self.item()
            def addpl(pl):
                if isinstance(pl, dict): e.update(pl)
            addpl(frame)
            if isinstance(hashtab, list):
                for x in hashtab: addpl(x)
            return e
        if t in (2, 6, 3, 5, 17, 239, 240):           # LIST, LANG, CLOS, PROM, DOT, ATTRLIST, ATTRLANG
            out = {}; keys = []; vals = []
            while True:
                attr = self.item() if has_attr else None
                tag = self.item() if has_tag else None
                car = self.item()
                keys.append(str(tag) if tag is not None else None); vals.append(car)
                flags = self.i32(); t2 = flags & 0xFF
                has_attr = bool(flags & (1 << 9)); has_tag = bool(flags & (1 << 10))
                if t2 == 254: break
                if t2 not in (2, 6, 3, 5, 17, 239, 240):
                    self._pushback(flags); tail = self.item(); break
            if all(k is not None for k in keys): return dict(zip(keys, vals))
            return vals
        if t in (22, 23):                             # EXTPTR, WEAKREF
            self.refs.append(None)
            if t == 22: self.item(); self.item()
            r = None
        elif t in (7, 8):                             # SPECIAL, BUILTIN
            n = self.i32(); return self.string(n)
        elif t == 9:                                  # CHARSXP
            n = self.i32()
            return None if n == -1 else self.string(n)
        elif t in (10, 13):                           # LGL, INT
            r = self.numvec(self.length(), ">i4", 4, np.int32)
        elif t == 14:                                 # REAL
            r = self.numvec(self.length(), ">f8", 8, np.float64)
        elif t == 15:                                 # CPLX
            n = self.length(); r = np.frombuffer(self.read(16 * n), dtype=">f8").reshape(-1, 2)
        elif t == 16:                                 # STR
            n = self.length(); r = [self.item() for _ in range(n)]
        elif t in (19, 20):                           # VEC, EXPR
            n = self.length(); r = [self.item() for _ in range(n)]
        elif t == 24:                                 # RAW
            n = self.length(); r = self.numvec(n, "u1", 1, np.uint8)
        elif t == 25:                                 # S4
            r = {"__S4__": True}
        elif t == 238:                                # ALTREP
            info = self.item(); state = self.item(); attr = self.item()
            cls = str(info[0]) if isinstance(info, list) else str(list(info.values())[0]) if isinstance(info, dict) else "?"
            r = self._altrep(cls, state)
            if isinstance(attr, dict) and attr: r = self._with_attr(r, attr)
            return r
        else:
            raise ValueError(f"unsupported SEXP type {t}")
        if has_attr:
            attr = self.item()
            r = self._with_attr(r, attr)
        return r

    def _pushback(self, flags):
        b = struct.pack(">i", flags)
        f = self.f
        class PB:
            def __init__(s): s.buf = b
            def read(s, n):
                if s.buf:
                    out = s.buf[:n]; s.buf = s.buf[n:]
                    if len(out) < n: out += f.read(n - len(out))
                    if not s.buf: self.f = f
                    return out
                return f.read(n)
        self.f = PB()

    def _altrep(self, cls, state):
        if cls in ("compact_intseq", "compact_realseq"):
            n, start, inc = (float(x) for x in np.asarray(state).ravel()[:3])
            arr = start + inc * np.arange(int(n))
            return arr.astype(np.int32 if cls == "compact_intseq" else np.float64)
        if cls.startswith("wrap_"):
            return state[0] if isinstance(state, list) else state
        if cls == "deferred_string":
            x = state[0] if isinstance(state, list) else (list(state.values())[0] if isinstance(state, dict) else state)
            return [str(v) for v in np.asarray(x).ravel()] if not isinstance(x, BigVec) else x
        return state

    def _with_attr(self, r, attr):
        if not isinstance(attr, dict): return r
        if isinstance(r, dict) and r.get("__S4__"):
            r.update(attr); return r
        names = attr.get("names")
        if isinstance(r, list) and isinstance(names, list) and len(names) == len(r):
            d = dict(zip([str(n) for n in names], r)); d["__attr__"] = {k: v for k, v in attr.items() if k != "names"}; return d
        return {"__value__": r, "__attr__": attr}

def read_rds(path, outdir):
    rd = Reader(path, outdir); ver = rd.header(); obj = rd.item()
    return obj, rd

def outline(o, depth=0, name="root", maxdepth=6, out=None):
    out = out if out is not None else []
    pad = "  " * depth
    if isinstance(o, dict):
        cls = o.get("class")
        cls = cls if isinstance(cls, list) else ""
        out.append(f"{pad}{name}: dict[{len(o)}] {cls if cls else ''}")
        if depth < maxdepth:
            for k, v in list(o.items())[:60]:
                if k in ("__S4__",): continue
                outline(v, depth + 1, k, maxdepth, out)
    elif isinstance(o, list):
        head = o[:3] if o and isinstance(o[0], (str, type(None))) else ""
        out.append(f"{pad}{name}: list[{len(o)}] {head}")
        if depth < maxdepth and o and not isinstance(o[0], (str, type(None))):
            for i, v in enumerate(o[:8]): outline(v, depth + 1, f"[{i}]", maxdepth, out)
    elif isinstance(o, np.ndarray):
        out.append(f"{pad}{name}: array{o.shape} {o.dtype} {o.ravel()[:4]}")
    else:
        out.append(f"{pad}{name}: {o!r}"[:200])
    return out

if __name__ == "__main__":
    import time
    t = time.time(); obj, rd = read_rds(sys.argv[1], sys.argv[2])
    print(f"read in {time.time()-t:.0f}s; big vectors: {rd.nbig}, {rd.bytes_big/1e9:.2f} GB on disk")
    print("\n".join(outline(obj)))
    pickle.dump(obj, open(os.path.join(sys.argv[2], "tree.pkl"), "wb"))
