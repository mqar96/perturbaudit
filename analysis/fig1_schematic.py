import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import sys
out=sys.argv[1]
fig,ax=plt.subplots(figsize=(7.2,4.0),dpi=300); ax.set_xlim(0,100); ax.set_ylim(0,56); ax.axis("off")
def box(x,y,w,h,t,fc,ec="#555",fs=6.5,bold=False,ha="center"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1",fc=fc,ec=ec,lw=0.7))
    tx=x+w/2 if ha=="center" else x+1.2
    ax.text(tx,y+h/2,t,ha=ha,va="center",fontsize=fs,weight="bold" if bold else None)
def arr(x1,y1,x2,y2,c="#555"): ax.annotate("",(x2,y2),(x1,y1),arrowprops=dict(arrowstyle="-|>",lw=0.7,color=c))
box(1,38,24,14,"Perturb-seq screen\ncells, guide calls,\nnon-targeting (NT) cells","#e8eef8")
box(1,4,24,14,"Inferred network\nany method;\nedge list or score matrix","#e8eef8")
box(1,20.5,24,15,"Internal controls\nNT cells → batch-matched decoys\ninterventions (knockdown z)\nindependent guides\nTSS positions, guide sequences","#f2f2f2",fs=5.6)
arr(13,38,13,35.7)
checks=[("1  Testable?","NT controls in parent's batches; child measured","untestable"),
        ("2a Supported?","|z(parent KD → child)| ≥ 3 or decoy q ≤ 0.05","unsupported*"),
        ("2b If unsupported: reversed?","child KD moves parent (|z| ≥ 5, q ≤ 0.05)","reversed"),
        ("3  Off-target?","guide seed match ≥ 11–12 nt at child promoter","off-target"),
        ("4  Neighbour?","parent and child TSSs < 10 kb apart","neighbour"),
        ("5  Guide-specific?","one guide moves child, the other does not","guide-specific"),
        ("6  Stable?","edge present in ≥ 50% of resampled networks","unstable")]
top=50; h=5.2; step=6.4; xc=34; wc=46
for i,(t,s,v) in enumerate(checks):
    y=top-i*step
    box(xc,y,wc,h,f"{t}  {s}","#ffffff",fs=5.4,ha="left")
    box(84,y+0.6,14,h-1.2,v,"#fbe3d6",ec="#c8865a",fs=5.6)
    arr(xc+wc+0.3,y+h/2,83.7,y+h/2,"#c8865a")
    if i<len(checks)-1: arr(xc+6,y,xc+6,y-step+h+0.1)
yend=top-(len(checks)-1)*step
box(xc,yend-7.5,wc,5,"pass all checks  →  audited edge","#d9ecd9",ec="#4a944a",fs=6.2,bold=True)
arr(xc+6,yend,xc+6,yend-2.4,"#4a944a")
arr(25,28,xc-0.4,top+h/2-2)
arr(25,11,xc-0.4,top+h/2-3.5)
ax.text(34,0.6,"Verdict = first failing check in this fixed order. *Untestable instead if the parent's own knockdown |z| < 2.",fontsize=5.3,color="#333")
plt.savefig(out,bbox_inches="tight"); print("ok")
