args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({library(Rcpp); library(dplyr); library(purrr); library(inspre)})
A <- Sys.getenv("ADAPRE_DIR", "repos/ADAPRE")
src <- readLines(file.path(A, "R", "adapre_inspre_core.R"))
if (Sys.getenv("ADAPRE_FIX", "1") == "1") {
  # Minimal fix: the public code passes the unfiltered beta_obs, so the adaptive penalty is silently skipped
  # (C++ checks length(beta_obs) == D) whenever filter_tce drops genes. Pass beta_obs for the kept rows only.
  f0 <- grep("^fit_inspre_from_X <- function", src)[1]
  hits <- grep("adaptive_lambda = adaptive_lambda, beta_obs = beta_obs$", src)
  i <- hits[hits > f0][1]
  src[i] <- sub("beta_obs = beta_obs$", "beta_obs = beta_obs[rownames(R_hat)]", src[i])
}
eval(parse(text = src))
Rcpp::sourceCpp(file.path(A, "src", "adapre_inspre_cpp.cpp"))
X <- as.matrix(data.table::fread(args[1], header = TRUE, check.names = FALSE))
targets <- scan(args[2], character(), quiet = TRUE)
targets[targets == "non-targeting"] <- "control"
res <- fit_inspre_from_X(X = X, targets = targets, weighted = TRUE, verbose = 0, constraint = "UV", adaptive_lambda = TRUE)
G <- res$G_hat
score <- matrix(0, dim(G)[1], dim(G)[2])
for (k in seq_len(dim(G)[3])) score <- score + (abs(G[, , k]) > 0)
score <- score + abs(G[, , dim(G)[3]]) / (max(abs(G[, , dim(G)[3]])) + 1e-12)
dimnames(score) <- dimnames(G)[1:2]
write.table(score, args[3], sep = "\t", quote = FALSE)
