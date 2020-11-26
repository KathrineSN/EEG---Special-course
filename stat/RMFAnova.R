
library(nlme)

N170.data <- read.csv("N170data.csv", header = TRUE)

N170.data <- N170.data[complete.cases(N170.data), ]


model <- lme(Avg..amplitude ~ Social.condition * Emotional.condition, random=~1|Subject, data = N170.data)
anova(model)


EPN.data <- read.csv("EPNdata.csv", header = TRUE)

EPN.data <- EPN.data[complete.cases(EPN.data), ]


model <- lme(Avg..amplitude ~ Social.condition * Emotional.condition, random=~1|Subject, data = EPN.data)
anova(model)

EarlyLPP.data <- read.csv("EarlyLPPdata.csv", header = TRUE)

EarlyLPP.data <- EarlyLPP.data[complete.cases(EarlyLPP.data), ]


model <- lme(Avg..amplitude ~ Social.condition * Emotional.condition, random=~1|Subject, data = EarlyLPP.data)
anova(model)

LateLPP.data <- read.csv("LateLPPdata.csv", header = TRUE)

LateLPP.data <- LateLPP.data[complete.cases(LateLPP.data), ]


model <- lme(Avg..amplitude ~ Social.condition * Emotional.condition, random=~1|Subject, data = LateLPP.data)
anova(model)










