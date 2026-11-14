"""Stream hg38 (UCSC) from stdin and write +/-W bp windows around every GENCODE v43 basic gene TSS (sense strand of
the genome, uppercase). Output: TSV gene_name, chrom, tss, strand, sequence."""
import sys, gzip, re, os
W = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
tss = {}
with gzip.open("data/annot/gencode.v43.basic.gtf.gz", "rt") as f:
    for line in f:
        if line.startswith("#"): continue
        c = line.split("\t", 9)
        if c[2] != os.environ.get("FEAT", "gene"): continue
        nm = re.search(r'gene_name "([^"]+)', c[8]).group(1)
        t = int(c[3]) if c[6] == "+" else int(c[4])
        L = tss.setdefault(c[0], [])
        if not any(x[0] == nm and abs(x[1] - t) < 50 for x in L[-30:]): L.append((nm, t, c[6]))
out = open("data/seedscan/promoters_%s_w%d.tsv" % (os.environ.get("FEAT", "gene"), W), "w")
def flush(ch, seq):
    if ch not in tss: return
    s = "".join(seq).upper(); n = 0
    for nm, t, st in tss[ch]:
        a, b = max(0, t - 1 - W), min(len(s), t + W)
        out.write(f"{nm}\t{ch}\t{t}\t{st}\t{s[a:b]}\n"); n += 1
    print(ch, len(s), n, flush=True)
ch, seq = None, []
for line in sys.stdin:
    if line.startswith(">"):
        if ch: flush(ch, seq)
        ch = line[1:].strip().split()[0]; seq = []
    elif ch in tss: seq.append(line.strip())
if ch: flush(ch, seq)
