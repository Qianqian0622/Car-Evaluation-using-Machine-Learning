rm(list = ls())

require(foreign)
require(ggplot2)
require(MASS)
require(Hmisc)
require(reshape2)

car_eva <- read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", header=F, stringsAsFactors=F)  # import string variables as characters.
colnames(car_eva) <- c("buying", "maint", "doors", "persons", "lug_boot", "safety", "class")
head(car_eva)

car_eva$buying <- factor(car_eva$buying, levels=c("low", "med", "high", "vhigh"), ordered=TRUE)
car_eva$maint <- factor(car_eva$maint, levels=c("low", "med", "high", "vhigh"), ordered=TRUE)
car_eva$doors <- factor(car_eva$doors, levels=c("2", "3", "4", "5more"), ordered=TRUE)
car_eva$persons <- factor(car_eva$persons, levels=c("2", "4", "more"), ordered=TRUE)
car_eva$lug_boot <- factor(car_eva$lug_boot, levels=c("small", "med", "big"), ordered=TRUE)
car_eva$safety <- factor(car_eva$safety, levels=c("low", "med", "high"), ordered=TRUE)
car_eva$class <- factor(car_eva$class, levels=c("unacc", "acc", "good", "vgood"), ordered=TRUE)

set.seed(100)
trainingRows <- sample(1:nrow(car_eva), 0.7 * nrow(car_eva))
trainingData <- car_eva[trainingRows, ]
testData <- car_eva[-trainingRows, ]

options(contrasts = c("contr.treatment", "contr.poly"))
library(MASS)
polrMod <- polr(class ~ safety + lug_boot + doors + buying + maint, data=trainingData , Hess=T)
summary(polrMod)
#The summary output in R gives us the estimated log-odds Coefficients of each of the predictor varibales shown in the Coefficients section of the output. The cut-points for the adjecent levels of the response variable shown in the Intercepts section of the output.

coeffs=coef(summary(polrMod))
p=pnorm(abs(coeffs[,"t value"]),lower.tail = F)*2
cbind(coeffs,"p value"= round(p,3))

exp(coef(polrMod))

predictedClass <- predict(polrMod, testData)  # predict the classes directly
head(predictedClass)

predictedScores <- predict(polrMod, testData, type="p")  # predict the probabilites
head(predictedScores)


a=table(testData$class, predictedClass)  # confusion matrix

1-mean(as.character(testData$class) != as.character(predictedClass))  # misclassification error
mse=0
for (i in 1:4) {
       for (j in 1:4) {
             mse=mse+a[i,j]*(i-j)^2
         }
}
mse=mse/nrow(testData)



#One of the assumptions underlying ordinal logistic (and ordinal probit) regression is that 
#the relationship between each pair of outcome groups is the same.

sf <- function(y) {
  c('Y>=1' = qlogis(mean(y >= 1)),
    'Y>=2' = qlogis(mean(y >= 2)),
    'Y>=3' = qlogis(mean(y >= 3)),
    'Y>=4' = qlogis(mean(y >= 4)))
}

(s <- with(car_eva, summary(as.numeric(class) ~ safety + lug_boot + doors + buying + maint, fun=sf)))

glm(I(as.numeric(apply) >= 2) ~ safety, family="binomial", data = car_eva)
glm(I(as.numeric(apply) >= 3) ~ safety, family="binomial", data = car_eva)
glm(I(as.numeric(apply) >= 4) ~ safety, family="binomial", data = car_eva)

s[, 5] <- s[, 5] - s[, 4]
s[, 4] <- s[, 4] - s[, 3]
s[, 3] <- s[, 3] - s[, 3]
s 

plot(s, which=1:3, pch=1:3, xlab='logit', main=' ', xlim=c(-3,0))


##K fold cross validation
rm(list = ls())

library(MASS)

car_eva <- read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", header=F, stringsAsFactors=F)  # import string variables as characters.
colnames(car_eva) <- c("buying", "maint", "doors", "persons", "lug_boot", "safety", "class")
head(car_eva)

car_eva$buying <- factor(car_eva$buying, levels=c("low", "med", "high", "vhigh"), ordered=TRUE)
car_eva$maint <- factor(car_eva$maint, levels=c("low", "med", "high", "vhigh"), ordered=TRUE)
car_eva$doors <- factor(car_eva$doors, levels=c("2", "3", "4", "5more"), ordered=TRUE)
car_eva$persons <- factor(car_eva$persons, levels=c("2", "4", "more"), ordered=TRUE)
car_eva$lug_boot <- factor(car_eva$lug_boot, levels=c("small", "med", "big"), ordered=TRUE)
car_eva$safety <- factor(car_eva$safety, levels=c("low", "med", "high"), ordered=TRUE)
car_eva$class <- factor(car_eva$class, levels=c("unacc", "acc", "good", "vgood"), ordered=TRUE)

set.seed(101) # Set Seed so that same sample can be reproduced in future also

# Now Selecting 50% of data as sample from total 'n' rows of the data
#sample <- sample.int(n = nrow(car_eva), size = floor(.70*nrow(car_eva)), replace = F)
#train <- car_eva[sample, ]
#test  <- car_eva[-sample, ]

car_eva <- car_eva[sample(nrow(car_eva)),]
folds <- cut(seq(1,nrow(car_eva)),breaks=10,labels=FALSE)

mse=rep(0,10)

#Perform 10 fold cross validation
for(i in 1:10){
  #Segement your data by fold using the which() function 
  
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- car_eva[testIndexes, ]
  trainData <- car_eva[-testIndexes, ]
  #Use the test and train data partitions however you desire...
  newlm <- polr(class ~ safety + lug_boot + doors + buying + maint, data=trainData , Hess=T)
  newpred <- predict(newlm, testData)
  a=table(testData$class, newpred)
  for (m in 1:4) {
    for (j in 1:4) {
      mse[i]=mse[i]+a[m,j]*(m-j)^2/173
    }
  } 
}
mean(mse)
