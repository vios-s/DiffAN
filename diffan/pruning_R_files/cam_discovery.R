library(mgcv)
library(CAM)

X <- read.csv(file='{PATH_DATA}', header=FALSE, sep=",")
set.seed(42)
estDAG <- CAM(X, scoreName = "SEMGAM", numCores = 1, output = TRUE, variableSel = FALSE, 
              pruning = TRUE, pruneMethod = selGam, pruneMethodPars = list(cutOffPVal = 0.001))
write.csv(as.matrix(estDAG), row.names = FALSE, file = '{PATH_RESULTS}')
write.table(as.matrix(estDAG), file = "m.txt", sep = " ", row.names = FALSE, col.names = FALSE)
