
library(nlme)

data <- read.csv("bigdata.csv", header = TRUE)

data <- data[complete.cases(data), ]


model <- lme(Avg..amplitude ~ Social.condition * Emotional.condition, random=~1|Subject, data = data)
anova(model)








