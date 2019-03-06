
####################################################
# Load test and validation results from components #
####################################################

loaddatafiles <- function(datadir){
  testfile <- paste(datadir,'test_results.csv',sep='')
  validationfile <- paste(datadir,'validation_results.csv',sep='')
  if ( file.exists(testfile) & file.exists(validationfile) ){
    return( list(read.csv(testfile), read.csv(validationfile)) )
  }
  else{
    stop(paste("Missing results files in", datadir))
  }
}

demo <- loaddatafiles('../demographics/')
demo.test <- demo[[1]]
demo.validation <- demo[[2]]

capt <- loaddatafiles('../captions/')
capt.test <- capt[[1]]
capt.validation <- capt[[2]]

desc <- loaddatafiles('../descriptions/')
desc.test <- desc[[1]]
desc.validation <- desc[[2]]

test <- data.frame(file=demo.test$file, 
                   truth=demo.test$truth, 
                   demographics_label=demo.test$label, 
                   demographics_prob=demo.test$prob, 
                   captions_label=capt.test$label,
                   captions_prob=capt.test$prob,
                   desc_label=desc.test$label,
                   desc_prob=desc.test$prob)

validation <- data.frame(file=demo.validation$file, 
                   truth=demo.validation$truth, 
                   demographics_label=demo.validation$label, 
                   demographics_prob=demo.validation$prob, 
                   captions_label=capt.validation$label,
                   captions_prob=capt.validation$prob,
                   desc_label=desc.validation$label,
                   desc_prob=desc.validation$prob)

#Tidy up
rm(demo, demo.test, demo.validation, capt, capt.test, capt.validation, desc, desc.test, desc.validation)


####################################################
# Helper functions for fitting model and display   #
####################################################


suppressMessages(library(e1071))

fit.svm <- function(trainset){
  svm.model <- svm(as.factor(truth) ~ ., data=trainset, kernel='radial', cost=8.0, gamma=2.0, probability=TRUE)
  return(svm.model)
}

test.svm <- function(svm.model, testset){
  svm.pred <- predict(svm.model, testset[,-1], type="response")
  return(svm.pred)
}

#Create confusion table, print stats.
confuse <- function(v1, v2){
	tab <- table(paste('predict',v1), v2)
	print(tab)
	freqs <- as.data.frame(tab)$Freq
	stats(freqs[4],freqs[2], freqs[3], freqs[1])
}

# Print minority-class precision, recall & f1, and overall accuracy.
stats <- function(tp, fp, fn, tn){ 
  precision <- tp/(tp+fp)
  recall <- tp/(tp+fn)
  f1 <- 2*((precision*recall)/(precision+recall))
  acc <- (tp+tn)/(tp+tn+fp+fn)
  cat(paste("precision:", round(precision,3), "\n"))
  cat(paste("recall:", round(recall,3), "\n"))
  cat(paste("f1:", round(f1,3), "\n"))
  cat(paste("acc:", round(acc,3), "\n"))
  return()
}


####################################################
# Fit, predict, display                            #
####################################################


ensemble <- fit.svm(test[,  names(test) != 'file'])
pred <- test.svm(ensm, validation)

cat("Validation SVM Ensemble\n")
confuse(pred, validation$truth)
