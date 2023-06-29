isotonic.maxflow <- function (C) 
{
    n <- nrow(C)
    F <- matrix(nrow = n, ncol = n, 0)
    ready <- FALSE
    labels <- matrix(nrow = n, ncol = 2, 0)
    while (!ready) {
        labels[, ] <- 0
        labels[1, ] <- c(1, 10000)
        queue <- c(1)
        index <- 1
        while (labels[n, 1] == 0) {
            if (length(queue) == 0) {
                return(labels)
            }
            v <- queue[index]
            queue <- queue[-index]
            index1 <- c(1:n)[C[v, ] > 0 & labels[, 1] == 0]
            index2 <- c(1:n)[C[, v] > 0 & labels[, 1] == 0]
            for (i in index1) {
                if (F[v, i] < C[v, i]) {
                  labels[i, 1] <- v
                  labels[i, 2] <- min(labels[v, 2], C[v, i] - 
                    F[v, i])
                  queue <- c(queue, i)
                }
            }
            for (i in index2) {
                if (F[i, v] > 0) {
                  labels[i, 1] <- v
                  labels[i, 2] <- min(labels[v, 2], F[i, v])
                  queue <- c(queue, i)
                }
            }
        }
        v <- n
        lambda <- labels[n, 2]
        while (v != 1) {
            u <- labels[v, 1]
            if (C[u, v] > 0) {
                F[u, v] <- F[u, v] + lambda
            }
            else {
                F[v, u] <- F[v, u] - lambda
            }
            v <- u
        }
    }
}


isotonic.monoreg <- function (ordermat, g, w, subset) 
# implements the algorithm of 1. Maxwell and Muckstadt, 2. Spouge, Wan and Wilbur, 3. Picard
# ordermat is an incidence matrix representing the partial order
# that is: ordermat[i,j] == 1 iff i <= j.
# g contains the unconstrained estimates to be made monotone
# w[i] is the weight of g[i], typically the number of observations on which the unconstrained
# estimate g[i] has been computed
{
    m <- length(subset)
    # trivial case: the subset only contains one element
    if (m == 1) {
        g.star.index <- subset
        g.star.value <- g[subset]
        blocks <- list(list(block = subset, av = g[subset]))
    }
    else {
        # compute the weighted average of g, with weights w, on the subset
        av <- sum(g[subset] * w[subset])/sum(w[subset])
        print(av)
        b <- w[subset] * (g[subset] - av)
        index.pos <- c(1:m)[b > 0]
        index.neg <- c(1:m)[b < 0]

        if (length(index.pos) == 0 && length(index.neg) == 0) {
            g.star.index <- subset
            g.star.value <- rep(av, m)
            blocks <- list(list(block = subset, av = av))
        }
        else {
            network <- matrix(nrow = m + 2, ncol = m + 2, 0)
            # add directed edge with "infinite" capacity from x to y iff x <= y.
            network[2:(m + 1), 2:(m + 1)] <- ordermat[subset,subset] * 1e+05
            
            # add edge from source to elements with b > 0 with capacity b
            network[1, index.pos + 1] <- b[index.pos]
            # add edge from elements with b < 0 to sink, with capacity -b
            network[index.neg + 1, m + 2] <- -b[index.neg]
            # solve the maximum flow problem on the constructed network to find the maximum upper set
            labels <- isotonic.maxflow(network)
            #print(sum(labels))
            temp <- as.vector(labels[2:(m + 1), 2])
            # determine "projection pair": the maximum upper set and minimum lower set
            index <- c(1:m)[temp > 0]
            upset <- subset[index]
            lowset <- subset[-index]
            
            if (length(upset) == 0) {
                g.star.index <- subset
                g.star.value <- rep(av, m)
                blocks <- list(list(block = subset, av = av))
            }
            else if (length(lowset) == 0) {
                g.star.index <- subset
                g.star.value <- rep(av, m)
                blocks <- list(list(block = subset, av = av))
            }
            else {
                # recurse on the maximum upperset U and minimum lower set L
                up <- isotonic.monoreg(ordermat, g, w, upset)
                low <- isotonic.monoreg(ordermat, g, w, lowset)
                # g*|U = (g|U)* and g*|L = (g|L)*
                g.star.index <- c(low$g.star.index, up$g.star.index)
                g.star.value <- c(low$g.star.value, up$g.star.value)
                blocks <- c(low$blocks, up$blocks)
            }
        }
    }
    list(g.star.index = g.star.index, g.star.value = g.star.value, 
        blocks = blocks)
}


isotonic.main <- function (ordermat, y, w = rep(1, length(y))) 
{
    n <- length(y)
    out <- isotonic.monoreg(ordermat, y, w, 1:n)
    out$g.star.value[order(out$g.star.index)]
}

#isotonic.main(ordermat.quasi2, c(0.4,0.2,0.6,0.4))

#library(reticulate)
#pd <- import("pandas")
ordermat<- pd$read_pickle("ordermat.pickle")
#pima = read.csv(file = 'pima.csv', header = F)
#y = pima$V9
res = isotonic.main(ordermat, y)
#sum = 0
#res
##for (g in res) {
#  sum = sum + min(1-g, g)
#}
#sum

