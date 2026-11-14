"""Genome-wide CRISPRi/a seed off-target maps for public libraries (resource).
For every guide: genes (other than the target) with a seed match >= MIN nt at an NGG site within +/-1 kb of any GENCODE v43
basic transcript TSS; flags pairs within 1 Mb of the target (neighbour check covers those); adds the knock-down rate
observed for that seed length in two genome-wide Perturb-seq screens (HCT116 Xaira, K562 GWPS; gene-column bottom 0.5%)."""
import sys, gzip, re, numpy as np, pandas as pd
sys.path.insert(0, "perturbaudit/src")
from perturbaudit import seedscan as ss
MIN = 10; OUT = "derived/offtarget_maps"; import os; os.makedirs(OUT, exist_ok=True)
L = "data/libs/"; T = "data/ScreenProcessing/library_tables/"
libs = {}
def wtab(f):
    x = pd.read_csv(T + f, sep="\t", dtype=str); x.columns = ["sgId"] + list(x.columns[1:])
    x = x[~x.gene.str.contains("negative_control|non-targeting", case=False, na=True)]
    return pd.DataFrame(dict(guide=x.sgId, target=x.gene, protospacer=x.sequence))
libs["hCRISPRi-v2 (Horlbeck 2016)"] = wtab("CRISPRi_v2_human_librarytable.txt")
libs["hCRISPRa-v2 (Horlbeck 2016)"] = wtab("CRISPRa_v2_human_librarytable.txt")
libs["hCRISPRi-v1 (Gilbert 2014)"] = wtab("CRISPRi_v1.txt")
x = pd.read_excel(L + "elife-81856-supp1-v2.xlsx", dtype=str)
x["gene"] = x.sgID_A.str.split("_").str[0]  # Excel turned some symbols (SEPT/MARCH) into dates; the sgID keeps the real name
x = x[~x.gene.str.contains("non-targeting|negative", case=False, na=True)]
libs["hCRISPRi-dual (Replogle 2022)"] = pd.concat([pd.DataFrame(dict(guide=x.sgID_A, target=x.gene, protospacer=x.protospacer_A)), pd.DataFrame(dict(guide=x.sgID_B, target=x.gene, protospacer=x.protospacer_B))])
h = pd.read_csv("data/hartman/perturb_seed_reproduce/data/xaira_hct116_2kb_filt_no_lfc_cutoff_off_target_candidates.csv.gz", low_memory=False, usecols=["guide_id", "gene", "protospacer_sequence"], dtype=str).dropna().drop_duplicates()
libs["Xaira X-Atlas/Orion (guides in Hartman table)"] = pd.DataFrame(dict(guide=h.guide_id + "::" + h.protospacer_sequence, target=h.gene, protospacer=h.protospacer_sequence))
for k, d in libs.items():
    d["protospacer"] = d.protospacer.str.upper().str[-20:]
    d.drop_duplicates(["guide", "protospacer"], inplace=True)
tss = {}
with gzip.open("data/annot/gencode.v43.basic.gtf.gz", "rt") as f:
    for line in f:
        if line.startswith("#"): continue
        c = line.split("\t", 9)
        if c[2] != "gene": continue
        tss.setdefault(re.search(r'gene_name "([^"]+)', c[8]).group(1), (c[0], int(c[3]) if c[6] == "+" else int(c[4])))
def near(a, b):
    u, v = tss.get(a), tss.get(b)
    return u is not None and v is not None and u[0] == v[0] and abs(u[1] - v[1]) < 1_000_000
rate = {10: (0.025, 0.016), 11: (0.093, 0.064), 12: (0.147, 0.104), 13: (0.174, 0.119)}
def kd(L_):
    r = rate.get(min(int(L_), 13)) if L_ >= 10 else None
    return r if r else (np.nan, np.nan)
S = ss.build_sites("data/seedscan/promoters_transcript_w1000.tsv")
summ = []
import gc, json
for name, d in libs.items():
    fn = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") + "_seed_offtargets_ge10.csv.gz"
    if os.path.exists(f"{OUT}/{fn}"): print("skip", name); continue
    parts = []
    for c0 in range(0, len(d), 40000):
        r = ss.scan(d.iloc[c0:c0 + 40000], S, min_len=MIN); parts.append(r[r.target != r.gene]); del r; gc.collect()
    R = pd.concat(parts); del parts; gc.collect()
    R["within_1Mb_of_target"] = [near(a, b) for a, b in zip(R.target, R.gene)]
    kk = np.array([kd(l) for l in R.seed_len]); R["kd_rate_HCT116"] = kk[:, 0]; R["kd_rate_K562_GWPS"] = kk[:, 1]
    R["guide"] = R.guide.str.split("::").str[0]
    R.sort_values(["seed_len", "guide"], ascending=[False, True]).to_csv(f"{OUT}/{fn}", index=False)
    g = R[~R.within_1Mb_of_target]
    summ.append(dict(library=name, guides=d.guide.nunique(), targets=d.target.nunique(), pairs_ge10=len(R), distant_ge11=int((g.seed_len >= 11).sum()), distant_ge12=int((g.seed_len >= 12).sum()),
                     guides_with_distant_ge12=g[g.seed_len >= 12].guide.nunique(), expected_distant_knockdowns_K562=round(float(np.nansum(g.kd_rate_K562_GWPS[g.seed_len >= 11])), 0), file=fn))
    print(summ[-1], flush=True); open(f"{OUT}/summary.jsonl", "a").write(json.dumps(summ[-1]) + "\n"); del R, g; gc.collect()
