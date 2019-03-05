library(e1071)
library(knitr) # Used for formatting script output only.
library(randomForest)
options(warn=-1)

# Presumed CSV locations
datadir <- "../data/"
trainfile <- paste(datadir,'train.csv',sep='')
testfile <- paste(datadir,'test.csv',sep='')
validationfile <- paste(datadir,'validation.csv',sep='')


##########################################################
# Data cleaning and loading functions                   #
##########################################################

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
	} else if (is.na(level)){
		return(NA)
	}
	return('other')
}


ethnise <- function(level){
	if (is.na(level)){
		return(NA)
	}
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


##########################################################
# Fitting, crossvalidation and performance display       #
##########################################################
# Naive Bayes
fit.nb <- function(trainset){
  nb.model <- naiveBayes(trainset[,-1], trainset$scam)
  return(nb.model)
}

test.nb <- function(nb.model, testset){
  results <- predict(nb.model, testset, type='raw')
  return(results[,2])
}


# Random Forests
fit.rf <- function(trainset){
  rf.model <- randomForest(scam ~ ., data=trainset, na.action=na.omit)
  return(rf.model)
}

test.rf <- function(rf.model, testset){
  rf.pred <- predict(rf.model, testset, type="response")
  return(rf.pred)
}


#Joint model class
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

getjointlabels <- function(big.joint.pred){
	missy <- big.joint.pred[,2]
	big.joint.pred <- big.joint.pred[,1]
	tcm <- mean(big.joint.pred[missy])
	fcm <- mean(big.joint.pred[!missy])
	labels <- ifelse(big.joint.pred > fcm, 1, 0)
	labels[missy] <- ifelse(big.joint.pred[missy] > tcm, 1, 0)
	return(labels)
}

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
	cat(label)

	#Produce a confusion matrix
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


# Load training data
train <- loaddata(trainfile)

# Crossvalidation
if ( 'fold' %in% names(train) ){
  k <- length(unique(train$fold))
  null <- crossvalidate(train, fit.joint, test.joint, paste(k,"-fold crossvalidation on training", sep=''))
}

# Fit model on training data
joint.model <- fit.joint(train[,! names(train) %in% c('fold','number')])

# Test
if (file.exists(testfile)){
  test <- loaddata(testfile)
  test$status <- factor(test$status, levels(train$status)) #Tweak for missing factor.
  test.pred <- test.joint(joint.model, test[,! names(test) %in% c('fold','number')])
  test.labels <- getjointlabels(test.pred)
  null <- show_performance(test.labels, "Test set performance", test$scam)
  test.results <- data.frame(file=test$number, truth=test$scam, label=test.labels, probs=test.pred[,1])
  test.results <- test.results[order(test.results$file),]
  write.csv(test.results, "test_results.csv", row.names=F)
}

# Validation
if (file.exists(validationfile)){
  validation <- loaddata(validationfile)
  validation$status <- factor(validation$status, levels(train$status)) #Tweak for missing factor.
  validation.pred <- test.joint(joint.model, validation[,! names(validation) %in% c('fold','number')])
  validation.labels <- getjointlabels(validation.pred)
  null <- show_performance(validation.labels, "Validation set performance", validation$scam)
  validation.results <- data.frame(file=validation$number, truth=validation$scam, label=validation.labels, probs=validation.pred[,1])
  validation.results <- validation.results[order(validation.results$file),]
  write.csv(validation.results, "validation_results.csv", row.names=F)
}




