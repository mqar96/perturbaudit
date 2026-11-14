args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({library(inspre); library(dplyr)})
R <- as.matrix(read.table(args[1], header = TRUE, sep = "\t", row.names = 1, check.names = FALSE))
SE <- as.matrix(read.table(args[2], header = TRUE, sep = "\t", row.names = 1, check.names = FALSE))
rho <- as.numeric(Sys.getenv("INSPRE_RHO", "10"))
W <- inspre::make_weights(SE)
lam <- if (length(args) >= 4) scan(args[4], numeric(), quiet = TRUE) else NULL
res <- inspre::fit_inspre_from_R(R, W = W, rho = rho, lambda = lam, nlambda = 20, verbose = 0, ncores = 1)
G <- res$G_hat
D <- dim(G)[1]
score <- matrix(0, D, D)
for (k in seq_len(dim(G)[3])) score <- score + (abs(G[, , k]) > 0)
score <- score + abs(G[, , dim(G)[3]]) / (max(abs(G[, , dim(G)[3]])) + 1e-12)
dimnames(score) <- dimnames(G)[1:2]
score[is.na(score)] <- 0
write.table(score, args[3], sep = "\t", quote = FALSE)
