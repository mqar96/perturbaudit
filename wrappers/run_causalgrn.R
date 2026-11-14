args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({library(CausalGRN); library(igraph)})
X <- as.matrix(data.table::fread(args[1]))
umi <- scan(args[2], numeric(), quiet = TRUE)
grp <- scan(args[3], character(), quiet = TRUE)
mode <- args[5]
grp[grp == "non-targeting"] <- "WT"
cnt <- round(expm1(pmax(X, 0)) * umi / 1e4)
rownames(cnt) <- paste0("c", seq_len(nrow(cnt))); colnames(cnt) <- colnames(X)
Y <- scale(log1p(cnt)); Y[is.na(Y)] <- 0
names(grp) <- rownames(cnt)
genes <- colnames(X)
if (mode == "causalgrn") {
  skel <- infer_skeleton(count = cnt, Y = Y, alpha = 0.05, min_abspcor = 0, ncores = 1)
  stat <- calc_perturbation_effect(Y = Y, group = grp, ncores = 1)
  g <- infer_causalgrn(graph = skel$graph, stat = stat, alpha = 0.05, max_order = 2)
} else {
  wt <- Y[grp == "WT", , drop = FALSE]
  tg <- setdiff(unique(grp), "WT"); tg <- tg[tg %in% genes]
  pts <- lapply(tg, function(t) Y[grp == t, , drop = FALSE]); names(pts) <- tg
  g <- run_gies(wt = wt, pts = pts)
}
A <- matrix(0, length(genes), length(genes), dimnames = list(genes, genes))
if (inherits(g, "igraph")) {
  el <- igraph::as_edgelist(g, names = TRUE)
  if (nrow(el)) A[el] <- 1
} else {
  A[] <- as.matrix(g)[genes, genes]
}
write.table(A, args[4], sep = "\t", quote = FALSE)
