args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({library(inspre); library(dplyr)})
X <- as.matrix(data.table::fread(args[1], header = TRUE, check.names = FALSE))
targets <- scan(args[2], character(), quiet = TRUE)
targets[targets == "non-targeting"] <- "control"
res <- inspre::fit_inspre_from_X(X, targets, rho = as.numeric(Sys.getenv("INSPRE_RHO", "10")), nlambda = 20, verbose = 0, ncores = 1, DAG = Sys.getenv("INSPRE_DAG", "FALSE") == "TRUE")
G <- res$G_hat
D <- dim(G)[1]
score <- matrix(0, D, D)
for (k in seq_len(dim(G)[3])) score <- score + (abs(G[, , k]) > 0)
score <- score + abs(G[, , dim(G)[3]]) / (max(abs(G[, , dim(G)[3]])) + 1e-12)
dimnames(score) <- dimnames(G)[1:2]
write.table(score, args[3], sep = "\t", quote = FALSE)
