# Classification

This page covers the current performance of the demographic classifier. 

I will only use age, gender, ethnicity, the normalised occupation, and the various location values. 

The first code block sets up the data preparation code. 


```r
library(e1071)
library(ROCR)
```

```
## Loading required package: gplots
```

```
## 
## Attaching package: 'gplots'
```

```
## The following object is masked from 'package:stats':
## 
##     lowess
```

```r
library(PRROC)
options(warn=-1)

employ <- function(level){
	if( grepl("army|milit|marine|soldier|captain|general|solda|force", level) ){
		return("military")
	} else if ( grepl("stud?ent|studi|coll", level) ){
		return("student")
	} else if ( grepl("self|own|independ|entre|freelanc|propia|autonomo", level)) {
		return("self-employed")
	} else if ( grepl("engin|ingenier|mechanic|mecanic|automot", level)) {
		return("engineering")
	} else if ( grepl("gover|civil|public|^un[$ ]", level)) {
		return("government")
	} else if ( grepl("academ|profes|research|lectur|universi|ologist|phd", level)){
		return("academic")
	} else if ( grepl("nurs|enfermer|care|trainer|nanny|baby|niñera|social", level)) {
		return("carer")
	} else if ( grepl("construc|carpent|roof|build|survey|ass?es|crane|equipment", level)){
		return("construction")
	} else if ( grepl("secur|detect|polic|investig|guard|custod|correct", level)){
		return("security")
	} else if ( grepl("econom|analy", level)){
		return("analyst")
	} else if ( grepl("farm|agri", level)){
		return("agriculture")
	} else if ( grepl("sail|sea|fish", level)){
		return("naval")
	} else if ( grepl("weld|factory|manufact|machin|industr", level)){
		return("manufacturing")
	} else if ( grepl("tech|inform|^it[$ ]|telecom|téch|software|sistem|system|tecnico|técnico|program|network|comput|electro|teck|develop", level) ){
		return("technology")
	} else if ( grepl("retail|comerci|shop|clerk|store|wait|vend|sell|cashier|assist|tender|customer|asist|mesero|restaur|camarer", level)){
		return("service")
	} else if ( grepl("tour|holiday|vacat|steward|flight|travel|turis|hotel", level)){
		return("tourism")
	} else if ( grepl("sale|market|ventas", level)) {
		return("sales")
	} else if ( grepl("writ|journal|period", level)){
		return("writing")
	} else if ( grepl("handy|repair|repare|maint|plumb|electr|manteni|hvac", level)){
		return("repair")
	} else if ( grepl("estat", level)){
		return("real estate")
	} else if ( grepl("teach|educa|docen|maestr|lehr", level)) {
		return("teacher")
	} else if ( grepl("manag|supervis", level)){
		return("manager")
	} else if ( grepl("contra", level)){
		return("contractor")
	} else if ( grepl("ama de casa|wife|mother|mom|home|hogar", level)){
		return("housewife")
	} else if ( grepl("unemploy|desempl|not work", level)){
		return("unemployed")
	} else if ( grepl("financ|bank|insur|trad|negoci|cajero", level)){
		return("finance")
	} else if ( grepl("chef|cook|bake|co[cs]iner|hospitali|food", level)){
		return("hospitality")
	} else if ( grepl("secret|admin|recep|office|human resources|clerical|profec|entry", level)){
		return("clerical")
	} else if ( grepl("driver|transport|deliver|ship|chofer|pilot|logist|cargo", level)){
		return("transport")	
	} else if ( grepl("housekeep|clean|limpi|janitor", level)){
		return("cleaner")
	} else if ( grepl("architec|arquitec", level)){
		return("architect")
	} else if ( grepl("account|contad", level)){
		return("accounting")
	} else if ( grepl("law|judge|solicitor|barrister|legal|attorney|abogad", level)){
		return("legal")
	} else if ( grepl("music|sport|play|produc|músico|deporti|conduc|soccer", level)){
		return("entertainment")
	} else if ( grepl("artist|art|paint|sculpt|boutique|photo|foto|choreo", level)){
		return("artist")
	} else if ( grepl("jewel|antiq|print", level)){
		return("specialist")
	} else if ( grepl("doctor|physic|ician|medic|psicolog|terap|therap|salud|health|médic|surgeon|denti|pharma", level)){
		return("medical")
	} else if ( grepl("beaut|styl|estili|peluquer|hair|salon|manic", level)){
		return("beauty")
	} else if ( grepl("fashion|model", level)){
		return("fashion")
	} else if ( grepl("design|decor|flower|desiñ|deisñ", level)){
		return("designer")
	} else if ( grepl("warehouse|work|opera|obrer|labor|labour|landscap|mining|mine|load|trabajo|pack|foreman", level)){
		return("manual")
	} else if ( grepl("bus[iy]?nes|empresa|execut|direct|ceo|ejecut", level)) {
		return("business")
	} else if ( grepl("consult", level)) {
		return("consultant")
	} else if ( grepl("retir|jubilad|pension", level)) {
		return("retired")
	} else if ( grepl("disab", level)) {
		return("disabled")
	} else if (is.na(level)) {
		return(NA)
	} else {
		return("other")
	}
 }

marry <- function(level){
	if (grepl('si[ng]{2}le', level)){
		return('single')
	} else if (grepl('win?doe?w', level)){
		return('widowed')
	} else if (grepl('married', level)){
		return('married')
	} else if (grepl('divorce', level)){
		return('divorced')
	} else if (grepl('sep[ae]?rat', level)){
		return('separated')
	} else if (grepl('relation|taken', level)){
		return('in relationship')
	}
	return('other')
}


ethnise <- function(level){
	if(! level %in% c("asian", "black", "hispanic", "middle eastern", "mixed", "native american", "pacific islander", "white")){
		return("other")
	}
	return(level)
 }


loaddata <- function(filename){
	rawdata <- read.csv(filename, na.strings=c('','NA'))
	data <- data.frame(scam=rawdata$scam, age=rawdata$age, gender=rawdata$gender, latitude=rawdata$latitude, longitude=rawdata$longitude, country=rawdata$country, fold=rawdata$fold, number=as.numeric(as.character(rawdata$number)))
	data$ethnicity <- as.factor(sapply(as.character(rawdata$ethnicity), ethnise))
	data$occupation <- as.factor(sapply(as.character(rawdata$occupation), employ))
	data$status <- as.factor(sapply(as.character(rawdata$status), marry))
	return(data)
}

train <- loaddata("newtrain.csv")
```




```r
crossvalidate <- function(features, fitmodel, testmodel, label){
  #Make results consistent
  set.seed(2017)

  trues <- c()
  preds <- c()
  missy <- c()
  numbs <- c()

  for(x in unique(features$fold)){
    #Train/test
    testindex <- which(features$fold == x)

    testset <- features[testindex,! names(features) %in% c('fold')]
    trainset <- features[-testindex,! names(features) %in% c('fold','number')]

    # Fit model
    model <- fitmodel(trainset)

    # Predict
    pred <- testmodel(model, testset[, ! names(testset) == 'number'])
    if (is.matrix(pred)){
	missy <- c(missy, pred[,2]) 
	pred <- pred[,1]
    }

    # Stash
    trues <- c(trues, testset$scam)
    preds <- c(preds, pred)
    numbs <- c(numbs, testset$number)
  }
  if (length(missy) < 2){
	  labels <- ifelse(preds > mean(preds), 1, 0)
  } else {
	tcm <- mean(preds[missy])
	fcm <- mean(preds[!missy])
	labels <- ifelse(preds > fcm, 1, 0)
	labels[missy] <- ifelse(preds[missy] > tcm, 1, 0)
  }
  null <- show_performance(labels, label, trues)
  matched <- cbind(numbs, labels, preds, trues)
  return(matched)
}



show_performance <- function(input.labels, label, truth){
#	input.eval <- prediction(input.pred, truth)

	#Produce a ROC plot.
#	input.roc <- performance(input.eval, 'tpr', 'fpr')
#	input.auc <- unlist(slot(performance(input.eval, 'auc'),'y.values'))
#	plot(input.roc, main=paste(label, "AUROC:", round(input.auc,3)), lwd=2, col='red')

	#Produce a P/R plot
#	posvals <- input.eval@predictions[[1]][truth == 1]
#	negvals <- input.eval@predictions[[1]][truth == 0]
#	input.pr <- pr.curve(scores.class0=posvals, scores.class1=negvals, curve=T)
#	input.pr.auc <- input.pr$auc.integral
#	plot(input.pr, main=paste(label, "AUPRC:", round(input.pr.auc,3)), auc.main=F, lwd=2, color='red')

	cat(label)

#	#Produce a confusion matrix
#	cutoffs <- input.eval@cutoffs[[1]]
#	input.acc <- unlist(slot(performance(input.eval, 'f'),'y.values'))
#	input.maxacc <- max(input.acc[!is.nan(input.acc)])
#	input.best <- cutoffs[which(input.acc == input.maxacc)]
#	given.labels <- ifelse(input.pred > input.best, "1", "0")
	confusion <- table(paste("predict",input.labels), truth)
	print(kable(confusion))
	cat("\n \n")
	
	#Calculate best precision, recall, f1
	confdf <- data.frame(confusion)	
	tn <- confdf$Freq[1]
	fp <- confdf$Freq[2]
	fn <- confdf$Freq[3]
	tp <- confdf$Freq[4]
	precision <- tp/(tp+fp)
	recall <- tp/(tp+fn)
	f1 <- 2*((precision*recall)/(precision+recall))
	accuracy <- (tp+tn)/(tp+tn+fp+fn)
	res <- data.frame(measure = c("precision","recall","f1","acc"), value = c(round(precision,3), round(recall,3), round(f1,3), round(accuracy, 3)))
	print(kable(res))
	cat("\n")
	return()
}


fit.nb <- function(trainset){
  nb.model <- naiveBayes(trainset[,-1], trainset$scam)
  return(nb.model)
}

test.nb <- function(nb.model, testset){
  results <- predict(nb.model, testset, type='raw')
  return(results[,2])
}
```

## Prior method: Lat/Lon 

My first approach is a simple Naive Bayesian model. The advantage here is the model is quick and easy to fit,
and robust to any missing data. 


```r
null <- crossvalidate(train[,! names(train) == 'country'], fit.nb, test.nb, "Naive Bayes")
```

Naive Bayes

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 6698|  543|
|predict 1 | 2238| 2597|

 


|measure   | value|
|:---------|-----:|
|precision | 0.537|
|recall    | 0.827|
|f1        | 0.651|
|acc       | 0.770|

We're fairly balanced here, with just slightly better recall than precision. Remember also that I looked at the 
subset for which all variables were present: 


```r
clean <- na.omit(train)
null <- crossvalidate(clean[,! names(clean) == 'country'], fit.nb, test.nb, "Naive Bayes (Subset)")
```

Naive Bayes (Subset)

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 3340|  421|
|predict 1 |  804| 2360|

 


|measure   | value|
|:---------|-----:|
|precision | 0.746|
|recall    | 0.849|
|f1        | 0.794|
|acc       | 0.823|

This gets us up to an F1 in the 70s. A binomial linear regression works about as well:


```r
fit.glm <- function(trainset){
  glm.model <- glm(scam ~ ., data=trainset, family="binomial")
  return(glm.model)
}


test.glm <- function(glm.model, testset){
  glm.pred <- predict(glm.model, testset, type="response")
  return(glm.pred)
}

null <- crossvalidate(clean[,! names(clean) == 'country'], fit.glm, test.glm, "LM (Subset)")
```

LM (Subset)

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 3385|  411|
|predict 1 |  759| 2370|

 


|measure   | value|
|:---------|-----:|
|precision | 0.757|
|recall    | 0.852|
|f1        | 0.802|
|acc       | 0.831|

## Variable Exploration

Time to try replacing lat/lon with the new country grouping.


```r
null <- crossvalidate(train[,! names(train) %in% c('latitude','longitude')], fit.nb, test.nb, "Naive Bayes (Country)")
```

Naive Bayes (Country)

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 6939|  504|
|predict 1 | 1997| 2636|

 


|measure   | value|
|:---------|-----:|
|precision | 0.569|
|recall    | 0.839|
|f1        | 0.678|
|acc       | 0.793|

Looks like country is adding a couple of percentage points over lat/lon. We could also try
adding in both of them.



```r
null <- crossvalidate(train, fit.nb, test.nb, "Naive Bayes (Country+Lat/Lon)")
```

Naive Bayes (Country+Lat/Lon)

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 6613|  461|
|predict 1 | 2323| 2679|

 


|measure   | value|
|:---------|-----:|
|precision | 0.536|
|recall    | 0.853|
|f1        | 0.658|
|acc       | 0.769|

Doesn't help. It might be worth comparing performance under different combinations generally.


```r
null <- crossvalidate(train[,c('scam','fold','number','age','gender')], fit.nb, test.nb, "Naive Bayes (Age+Gender)")
```

Naive Bayes (Age+Gender)

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 3835| 1173|
|predict 1 | 5101| 1967|

 


|measure   | value|
|:---------|-----:|
|precision | 0.278|
|recall    | 0.626|
|f1        | 0.385|
|acc       | 0.480|

```r
null <- crossvalidate(train[,c('scam','fold','number','age','gender','occupation')], fit.nb, test.nb, "Naive Bayes (Age+Gender+Occupation)")
```

Naive Bayes (Age+Gender+Occupation)

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 6192|  950|
|predict 1 | 2744| 2190|

 


|measure   | value|
|:---------|-----:|
|precision | 0.444|
|recall    | 0.697|
|f1        | 0.542|
|acc       | 0.694|

```r
null <- crossvalidate(train[,c('scam','fold','number','age','gender','ethnicity')], fit.nb, test.nb, "Naive Bayes (Age+Gender+Ethnicity)")
```

Naive Bayes (Age+Gender+Ethnicity)

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 4380|  428|
|predict 1 | 4556| 2712|

 


|measure   | value|
|:---------|-----:|
|precision | 0.373|
|recall    | 0.864|
|f1        | 0.521|
|acc       | 0.587|

```r
null <- crossvalidate(train[,c('scam','fold','number','age','gender','occupation','country')], fit.nb, test.nb, "Naive Bayes (Age+Gender+Occupation+Country)")
```

Naive Bayes (Age+Gender+Occupation+Country)

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 6082|  555|
|predict 1 | 2854| 2585|

 


|measure   | value|
|:---------|-----:|
|precision | 0.475|
|recall    | 0.823|
|f1        | 0.603|
|acc       | 0.718|

```r
null <- crossvalidate(train[,c('scam','fold','number','age','gender','occupation','ethnicity')], fit.nb, test.nb, "Naive Bayes (Age+Gender+Occupation+Ethnicity)")
```

Naive Bayes (Age+Gender+Occupation+Ethnicity)

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 5873|  642|
|predict 1 | 3063| 2498|

 


|measure   | value|
|:---------|-----:|
|precision | 0.449|
|recall    | 0.796|
|f1        | 0.574|
|acc       | 0.693|

```r
final <- crossvalidate(train[,c('scam','fold','number','age','gender','occupation','ethnicity','country')], fit.nb, test.nb, "Naive Bayes (Age+Gender+Occupation+Ethnicity+Country)")
```

Naive Bayes (Age+Gender+Occupation+Ethnicity+Country)

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 6518|  490|
|predict 1 | 2418| 2650|

 


|measure   | value|
|:---------|-----:|
|precision | 0.523|
|recall    | 0.844|
|f1        | 0.646|
|acc       | 0.759|

It looks like it's dubious how much value the location information is adding. A lot of the power it
provides might actually be given by the ethnicity variable, which suggests that the national information is
largely just helping exclude the 'hispanic' group.



## Other Classifiers



```r
null <- crossvalidate(clean[,! names(clean) %in% c('longitude','latitude')], fit.nb, test.nb, "NB (Subset, Country)")
```

NB (Subset, Country)

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 3411|  390|
|predict 1 |  733| 2391|

 


|measure   | value|
|:---------|-----:|
|precision | 0.765|
|recall    | 0.860|
|f1        | 0.810|
|acc       | 0.838|

```r
noloc <- train[,c('scam','fold','number','age','gender','occupation','ethnicity')]
cleannoloc <- na.omit(noloc)
```


```r
fit.svm <- function(trainset){
  svm.model <- svm(scam ~ ., data=trainset)
  return(svm.model)
}

test.svm <- function(svm.model, testset){
  svm.pred <- predict(svm.model, testset[,-1], type="response")
  return(svm.pred)
}

#null <- crossvalidate(clean[,! names(clean) %in% c('longitude','latitude')], fit.svm, test.svm, "SVM (Subset, Country)")
#null <- crossvalidate(clean[,! names(clean) %in% c('country')], fit.svm, test.svm, "SVM (Subset, Lon/Lat)")
```


```r
library(C50)

fit.dt <- function(trainset){
 dt.model <- C5.0(trainset[,-1], as.factor(trainset$scam))
 return(dt.model)
}

test.dt <- function(dt.model, testset){
 dt.pred <- predict(dt.model, testset, type='prob')
 return(dt.pred[,2])
}

null <- crossvalidate(cleannoloc, fit.dt, test.dt, "C5.0 (Subset, Basic)")
```

C5.0 (Subset, Basic)

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 3863|  627|
|predict 1 |  585| 2274|

 


|measure   | value|
|:---------|-----:|
|precision | 0.795|
|recall    | 0.784|
|f1        | 0.790|
|acc       | 0.835|

```r
null <- crossvalidate(clean[,! names(clean) %in% c('country')], fit.dt, test.dt, "C5.0 (Subset, Lat/Lon)")
```

C5.0 (Subset, Lat/Lon)

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 3753|  471|
|predict 1 |  391| 2310|

 


|measure   | value|
|:---------|-----:|
|precision | 0.855|
|recall    | 0.831|
|f1        | 0.843|
|acc       | 0.876|

The decision trees are working better than the Naive Bayes model! (within the subset where all data is available)
The C5.0 library falls over on the country names, so to try it on that feature I need to fiddle with its representation:


```r
clean$country <- as.factor(as.numeric(clean$country))
null <- crossvalidate(clean[,! names(clean) %in% c('latitude','longitude')], fit.dt, test.dt, "C5.0 (Subset, Country)")
```

C5.0 (Subset, Country)

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 3772|  469|
|predict 1 |  372| 2312|

 


|measure   | value|
|:---------|-----:|
|precision | 0.861|
|recall    | 0.831|
|f1        | 0.846|
|acc       | 0.879|

```r
null <- crossvalidate(clean, fit.dt, test.dt, "C5.0 (Subset, All)")
```

C5.0 (Subset, All)

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 3763|  457|
|predict 1 |  381| 2324|

 


|measure   | value|
|:---------|-----:|
|precision | 0.859|
|recall    | 0.836|
|f1        | 0.847|
|acc       | 0.879|

To summarise, including both location features doesn't help, but including one or the other creates a good classifier, with
both precision and recall over the 0.8 mark, and the highest subset accuracy yet. Given this, it seems worth exploring Random Forests.


```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```r
fit.rf <- function(trainset){
  rf.model <- randomForest(scam ~ ., data=trainset, na.action=na.omit)
  return(rf.model)
}

test.rf <- function(rf.model, testset){
  rf.pred <- predict(rf.model, testset, type="response")
  return(rf.pred)
}


null <- crossvalidate(cleannoloc, fit.rf, test.rf, "RandomForest (Subset, Basic)")
```

RandomForest (Subset, Basic)

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 3693|  420|
|predict 1 |  755| 2481|

 


|measure   | value|
|:---------|-----:|
|precision | 0.767|
|recall    | 0.855|
|f1        | 0.809|
|acc       | 0.840|

```r
null <- crossvalidate(clean[,! names(clean) %in% c('country')], fit.rf, test.rf, "RandomForest (Subset, Lat/Lon)")
```

RandomForest (Subset, Lat/Lon)

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 3765|  336|
|predict 1 |  379| 2445|

 


|measure   | value|
|:---------|-----:|
|precision | 0.866|
|recall    | 0.879|
|f1        | 0.872|
|acc       | 0.897|

A new high for accuracy and f1 within the subset. We're up to 0.88 recall, which is pretty strong for these variables.
I think it's time to combine models.


```r
setClass("JointModel", slots=c( "forest", "bayes"))

fit.joint <- function(trainset){
  rf.model <- fit.rf(trainset[,! names(trainset) == 'country'])
  nb.model <- naiveBayes(trainset[, !names(trainset) %in% c("scam","country","longitude","latitude")], trainset$scam)
  return(new("JointModel", forest=rf.model, bayes=nb.model))
}

test.joint <- function(joint.model, testset){
  rf.model <- joint.model@forest
  nb.model <- joint.model@bayes
  rf.pred <- predict(rf.model, testset, type="response")
  nb.pred <- predict(nb.model, testset, type="raw")[,2]
  joint.pred <- rf.pred
  missing <- is.na(rf.pred)
  joint.pred[missing] <- nb.pred[missing]
  return(cbind(joint.pred, missing))
}


null <- crossvalidate(train, fit.joint, test.joint, "RF+NB")
```

RF+NB

|          |    0|    1|
|:---------|----:|----:|
|predict 0 | 8390|  544|
|predict 1 |  546| 2596|

 


|measure   | value|
|:---------|-----:|
|precision | 0.826|
|recall    | 0.827|
|f1        | 0.826|
|acc       | 0.910|


```r
#Save my work.
df <- data.frame(file=as.numeric(as.character(null[,1])), demographics=null[,2], truth=null[,3])
df <- df[with(df, order(file)),]
write.csv(df, "trainlabels.csv", row.names=F)
```



```r
getjointlabels <- function(big.joint.pred){
	missy <- big.joint.pred[,2]
	big.joint.pred <- big.joint.pred[,1]
	tcm <- mean(big.joint.pred[missy])
	fcm <- mean(big.joint.pred[!missy])
	labels <- ifelse(big.joint.pred > fcm, 1, 0)
	labels[missy] <- ifelse(big.joint.pred[missy] > tcm, 1, 0)
	return(labels)
}

train <- loaddata("newtrain.csv")
big.joint.model <- fit.joint(train[,! names(train) %in% c('fold','number')])
test <- loaddata("newtest.csv")
test.pred <- test.joint(big.joint.model, test[,! names(test) %in% c('fold','number')])

test <- loaddata("newvalidation.csv")
big.joint.pred <- test.joint(big.joint.model, test[,! names(test) %in% c('fold','number')])
labels <- getjointlabels(big.joint.pred)
null <- show_performance(labels, "NB+RF :: Test", test$scam)
```

NB+RF :: Test

|          |    0|   1|
|:---------|----:|---:|
|predict 0 | 2725| 196|
|predict 1 |  149| 903|

 


|measure   | value|
|:---------|-----:|
|precision | 0.858|
|recall    | 0.822|
|f1        | 0.840|
|acc       | 0.913|

```r
null <- cbind(test$number, labels, test$scam)
df <- data.frame(file=as.numeric(as.character(null[,1])),  truth=null[,3], demographics=null[,2], probs=big.joint.pred)
df <- df[with(df, order(file)),]
write.csv(df, "validationlabels.csv", row.names=F)
```
