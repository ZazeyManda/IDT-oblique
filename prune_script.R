library(rpart)
library(rpart.plot)
boston.dat <- read.csv("boston.csv", header=T)
boston.tree <- rpart(medv ~.,data=boston.dat,cp=0,minbucket=1,minsplit=2)

table <- printcp(boston.tree)

boston.pruned <- prune(boston.tree,cp=3.3758e-02 )
rpart.plot(boston.pruned,nn=TRUE )
bank.pruned2 <- prune(bank.tree,cp=0.005) # Behoort tot CP9.pdf
rpart.plot(bank.pruned2, nn=TRUE)

# 7 
get_node_data <- function(tree = bank.pruned2, node){
  rule <- path.rpart(tree, node)
  rule_2 <- sapply(rule[[1]][-1], function(x) strsplit(x, '(?<=[><=])(?=[^><=])|(?<=[^><=])(?=[><=])', perl = TRUE))
  ind <- apply(do.call(cbind, lapply(rule_2, function(x) eval(call(x[2], bank.dat[,x[1]], as.numeric(x[3]))))), 1, all)
  bank.dat[ind,]
}

total = 1372
get_node_data(node = 7)
length(get_node_data(node = 7)$class) 

########### Feature selection + linear regression ###################
library(MASS)
auto.dat = read.csv('datasets/regression/Computer.csv')
summary(auto.dat)
#auto.dat$horsepower = as.numeric(as.factor(auto.dat$horsepower))
auto.lm <- lm(class ~ SPEED+HARD_DRIVE+RAM+SCREEN+CD+MULTI+FIRM+ADS+TREND, data=auto.dat)
summary(auto.lm)
