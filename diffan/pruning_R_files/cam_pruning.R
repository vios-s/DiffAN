library(mgcv)

source("diffan/pruning_R_files/train_gam.R", chdir=T)
source("diffan/pruning_R_files/selGam.R", chdir=T)
source("diffan/pruning_R_files/pruning.R", chdir=T)


dataset <- read.csv(file='{PATH_DATA}', header=FALSE, sep=",")
dag <- read.csv(file='{PATH_DAG}', header=FALSE, sep=",")
set.seed(42)
pruned_dag <- pruning(dataset, dag, pruneMethod = selGam, pruneMethodPars = list(cutOffPVal = {CUTOFF}, numBasisFcts = 10), output={VERBOSE})

write.csv(as.matrix(pruned_dag), row.names = FALSE, file = '{PATH_RESULTS}')
