"""Genome-wide seed-match scan for CRISPRi off-targets (no external aligner).
For every guide protospacer (20 nt) and every gene promoter window, report the longest PAM-proximal contiguous
match at an NGG-adjacent site on either strand ("seed length", as in Hartman et al.). Only matches >= min_len kept.
build_sites(promoter_tsv) -> sites; scan(guides, sites, min_len) -> DataFrame(guide, gene, seed_len)."""
import numpy as np, pandas as pd
_E = np.full(256, 255, np.uint8)
for i, b in enumerate(b"ACGT"): _E[b] = i
_RC = bytes.maketrans(b"ACGTN", b"TGCAN")

def _encode20(arr):  # arr: (n, 20) uint8 codes -> uint64, first base in the high bits
    v = np.zeros(len(arr), np.uint64)
    for k in range(20): v = (v << np.uint64(2)) | arr[:, k].astype(np.uint64)
    return v

def build_sites(promoter_tsv):
    genes, codes, gidx = [], [], []
    for gi, line in enumerate(open(promoter_tsv)):
        nm, ch, t, st, seq = line.rstrip("\n").split("\t"); genes.append(nm)
        for s in (seq.encode(), seq.encode().translate(_RC)[::-1]):
            a = np.frombuffer(s, np.uint8)
            if len(a) < 23: continue
            gg = np.flatnonzero((a[21:] == ord("G")) & (a[22:] == ord("G") if False else (np.r_[a[22:], 0][: len(a) - 21] == ord("G"))))
            gg = gg[gg + 23 <= len(a)]
            if not len(gg): continue
            win = np.lib.stride_tricks.sliding_window_view(a, 20)[gg]
            c = _E[win]; ok = (c != 255).all(1)
            codes.append(_encode20(c[ok])); gidx.append(np.full(ok.sum(), gi, np.int32))
    v = np.concatenate(codes); g = np.concatenate(gidx)
    o = np.argsort(v & np.uint64(0xFFFF), kind="stable")
    return dict(v=v[o], g=g[o], key=(v[o] & np.uint64(0xFFFF)).astype(np.int64), genes=np.array(genes, dtype=object))

def seed_len(x):  # x = xor of two encodings; contiguous equal bases from the 3' (PAM-proximal) end
    out = np.full(len(x), 20, np.int64); nz = x != 0
    y = x[nz]; low = y & (~y + np.uint64(1))  # lowest set bit
    out[nz] = (np.log2(low.astype(np.float64)).astype(np.int64)) // 2
    return out

def scan(guides, sites, min_len=8):
    """guides: DataFrame with columns guide, target, protospacer (20 nt)."""
    key = sites["key"]; starts = np.searchsorted(key, np.arange(65537))
    rows = []
    for gname, tgt, ps in guides[["guide", "target", "protospacer"]].dropna().itertuples(index=False):
        c = _E[np.frombuffer(ps.upper().encode(), np.uint8)]
        if len(c) != 20 or (c == 255).any(): continue
        q = _encode20(c[None, :])[0]; k = int(q & np.uint64(0xFFFF))
        a, b = starts[k], starts[k + 1]
        if a == b: continue
        L = seed_len(sites["v"][a:b] ^ q); m = L >= min_len
        if not m.any(): continue
        d = pd.DataFrame(dict(g=sites["g"][a:b][m], L=L[m])).groupby("g").L.max()
        for gi, l in d.items(): rows.append((gname, tgt, sites["genes"][gi], int(l)))
    R = pd.DataFrame(rows, columns=["guide", "target", "gene", "seed_len"])
    return R.groupby(["guide", "target", "gene"], as_index=False).seed_len.max()


def offtarget_map_from_guides(guides, sites, min_len=12, exclude=None):
    """Seed-match off-target map {target: {gene, ...}} from guide protospacers (columns guide, target, protospacer).
    exclude(target, gene) -> True drops a pair (e.g. genes within 1 Mb, which the neighbour check covers)."""
    R = scan(guides, sites, min_len=min_len)
    m = {}
    for t, g in zip(R.target, R.gene):
        if t == g or (exclude is not None and exclude(t, g)): continue
        m.setdefault(t, set()).add(g)
    return m


def promoter_windows(fasta, gtf, out_tsv, window=1000, feature="transcript"):
    """Write +/-window bp around every TSS in a GENCODE-style GTF(.gz) from a genome FASTA(.gz) (streamed, chromosome by
    chromosome; never holds more than one chromosome). Output TSV: gene, chrom, tss, strand, sequence -> build_sites()."""
    import gzip, re
    op = lambda p: gzip.open(p, "rt") if str(p).endswith(".gz") else open(p)
    tss = {}
    with op(gtf) as f:
        for line in f:
            if line.startswith("#"): continue
            c = line.split("\t", 9)
            if len(c) < 9 or c[2] != feature: continue
            m = re.search(r'gene_name "([^"]+)', c[8])
            if not m: continue
            t = int(c[3]) if c[6] == "+" else int(c[4]); L = tss.setdefault(c[0], [])
            if not any(x[0] == m.group(1) and abs(x[1] - t) < 50 for x in L[-30:]): L.append((m.group(1), t, c[6]))
    n = 0
    with op(fasta) as f, open(out_tsv, "w") as out:
        def flush(ch, seq):
            nonlocal n
            if ch not in tss: return
            s_ = "".join(seq).upper()
            for nm, t, st in tss[ch]:
                out.write(f"{nm}\t{ch}\t{t}\t{st}\t{s_[max(0, t - 1 - window):t + window]}\n"); n += 1
        ch, seq = None, []
        for line in f:
            if line.startswith(">"):
                if ch: flush(ch, seq)
                ch = line[1:].split()[0]; seq = []
            elif ch in tss: seq.append(line.strip())
        if ch: flush(ch, seq)
    return n


def load_offtarget_map(path, min_len=11, exclude_neighbours=True):
    """Read a perturbaudit seed off-target map (guide,target,gene,seed_len,within_1Mb_of_target,...) into the
    {target: {gene,...}} form audit_network(offtarget_map=...) expects."""
    import pandas as pd
    m = {}
    for ch in pd.read_csv(path, chunksize=1_000_000):
        ch = ch[ch.seed_len >= min_len]
        if exclude_neighbours and "within_1Mb_of_target" in ch: ch = ch[~ch.within_1Mb_of_target.astype(bool)]
        for t, g in zip(ch.target.astype(str), ch.gene.astype(str)): m.setdefault(t, set()).add(g)
    return m
