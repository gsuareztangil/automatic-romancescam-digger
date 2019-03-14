processors = 4
feature_selection = None #classifier_name  #None

classifiers = { 'SupportVectorMachine': 'SVM', 
				'RandomForests': 'RF', 
				'LinearSVM': 'LSVM', 
				'ExtraTreeForests': 'XTREE'}

classifier_name = classifiers['SupportVectorMachine'] # Classifier

estimators = { 'SupportVectorRegression': 'SVR',
			   'LinearSupportVectorRegression': 'LSVR',
			   'LogisticRegression': 'LR',
			   }

estimator_name = estimators['LogisticRegression']