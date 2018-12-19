rm(list=ls())  
ls() 
require(InformationValue)
require(e1071)
require(ROCR)
require(MLmetrics)
require(ModelMetrics)
require(rpart)
require(caret)
require(data.table)

setwd("./")


data<-read.csv('Rumi_01_19_2017.csv')

data2<-data

data2$CC<-tolower(data2$CC)

data2<-data2[(data2$CC %like% 'flu') | (data2$CC %like% 'fever') | (data2$CC %like% 'cough') |(data2$CC %like% 'chil') | (data2$CC %like% 'muscl') | (data2$CC %like% 'throat'),]

data2<-data2[data2$feverchills==1 | data2$Sorethroat==1 | data2$Cough==1 | data2$Muscleache==1 | data2$Headache==1 | data2$Fatigue==1 | data2$Vomit ==1 | data2$Nausea==1 | data2$Diarrhea==1,]

data3<-data2[,c('Flu','feverchills','Sorethroat','Cough','Muscleache','Headache','Fatigue','Vomit','Nausea','Diarrhea')]

data3$male<-0
data3$male[data2$GENDER=='MALE']<-1

data3$age.0_4<-0
data3$age.0_4[as.numeric(as.character(data2$AGE_AT_ARRIVAL))<=4]<-1

data3$age.5_15<-0
data3$age.5_15[as.numeric(as.character(data2$AGE_AT_ARRIVAL))<=15 & as.numeric(as.character(data2$AGE_AT_ARRIVAL))>=5]<-1

data3$age.16_44<-0
data3$age.16_44[as.numeric(as.character(data2$AGE_AT_ARRIVAL))<=44 & as.numeric(as.character(data2$AGE_AT_ARRIVAL))>=16]<-1

data3$age.45_64<-0
data3$age.45_64[as.numeric(as.character(data2$AGE_AT_ARRIVAL))<=64 & as.numeric(as.character(data2$AGE_AT_ARRIVAL))>=45]<-1

data3$age.65<-0
data3$age.65[as.numeric(as.character(data2$AGE_AT_ARRIVAL))>=65]<-1

write.csv(data3,'nyumc_new.csv')

