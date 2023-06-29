make.order <- function(x) {
    n <- nrow(x)
    ordermat <- matrix(nrow = n, ncol = n, 0)
    for (i in 1:(n - 1)) {
        for (j in (i + 1):n) {
            if (all(x[i, ] <= x[j, ])) {
                ordermat[i, j] <- 1
            }
            if (all(x[j, ] <= x[i, ])) {
                ordermat[j, i] <- 1
            }
        }
    }
    ordermat
}

multi.propp.wilson2 <- function(order, nrep, k) {
    n <- nrow(order)
    coalesced <- vector(length = nrep)
    labelings <- matrix(0, nrep, n)
    downsets <- vector("list", n)
    upsets <- vector("list", n)
    for (j in 1:nrep) {
        u_t <- runif(1)
        x_t <- sample(1:n, size = 1, replace = TRUE)
        t <- 1
        while (coalesced[j] != TRUE) {
            chain1 <- rep(k, n)
            chain2 <- rep(1, n)
            for (i in 1:t) {
                u <- u_t[i]
                x <- x_t[i]
                if (length(downsets[[x]]) == 0) {
                  downsets[[x]] <- downset(order, x)
                }
                if (length(upsets[[x]]) == 0) {
                  upsets[[x]] <- upset(order, x)
                }
                down <- downsets[[x]]
                up <- upsets[[x]]
                if (u > 0.5) {
                  chain1 <- decrease_label_nochain(x, down, chain1)
                  chain2 <- decrease_label_nochain(x, down, chain2)
                } else {
                  chain1 <- increase_label_nochain(x, up, chain1, 
                    k)
                  chain2 <- increase_label_nochain(x, up, chain2, 
                    k)
                }
                if (all(chain1 == chain2)) {
                  coalesced[j] <- TRUE
                }
            }
            t <- 2 * t
            u_t <- c(runif(t/2), u_t)
            x_t <- c(sample(1:n, size = t/2, replace = TRUE), 
                x_t)
        }
        labelings[j, ] <- chain1
    }
    list(labeling = labelings)
}

decrease_label_nochain <- function(x, down, list) {
    if (list[x] > 1 & !any(list[down] == list[x])) {
        list[x] <- list[x] - 1
    }
    list
}

downset <- function(ordermat, point) {
    downset <- seq(along = 1:ncol(ordermat))[ordermat[, point] == 
        1]
    downset
}

increase_label_nochain <- function(x, up, list, k) {
    if (list[x] < k & !any(list[up] == list[x])) {
        list[x] <- list[x] + 1
    }
    list
}

upset <- function (ordermat, point) {
    upset <- seq(along = 1:nrow(ordermat))[ordermat[point, ] == 
        1]
    upset
}

pmulti.propp.wilson2 <- function (ordermat, nrep, k) 
{
    ordermat <<- get("ordermat")
    dep <- c("ordermat", "multi.propp.wilson2", "downset", "upset", 
        "increase_label_nochain", "decrease_label_nochain")
    cores <- detectCores()
    cl <- makePSOCKcluster(cores)
    setDefaultCluster(cl)
    clusterExport(NULL, dep)
    work.div <- seq(floor(nrep/cores), floor(nrep/cores), length.out = cores)
    work.div[cores] <- work.div[cores] + nrep%%cores
    res <- parLapply(NULL, work.div, function(n) multi.propp.wilson2(ordermat, 
        n, k))
    stopCluster(cl)
    rm(ordermat)
    return(do.call("rbind", do.call("rbind", res)))
}
