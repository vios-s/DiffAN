selGam <-
function(X,pars = list(cutOffPVal = 0.001, numBasisFcts = 10),output = FALSE,k)
{
    result <- list()
    p <- dim(as.matrix(X))
    if(p[2] > 1)
    {
        selVec <- rep(FALSE, p[2])
        mod_gam <- train_gam(X[,-k],as.matrix(X[,k]),pars)
        pValVec <- summary.gam(mod_gam$model)$s.pv
        if(output)
        {
            cat("vector of p-values:", pValVec, "\n")
        }
        if(length(pValVec) != length(selVec[-k]))
        {
            show("This should never happen (function selGam).")
        }
        selVec[-k] <- (pValVec < pars$cutOffPVal)
    } else
    {
        selVec <- list()
    }
    return(selVec)
}
