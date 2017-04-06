library(xgboost, warn.conflicts = FALSE)
library(dplyr, warn.conflicts = FALSE)
library(lubridate, warn.conflicts = FALSE)

data <- read.csv("/home/ishan/Desktop/Projects/predicting-scope-of-stackoverflow-questions/combined.csv", header=TRUE)
head(data,3)
unique(data$Reason)
sapply(data,class)

set.seed(7)

#Sample Indexes
dt = sample(1:nrow(data), size=0.2*nrow(data))

#Split data

train <- data[-dt,]
dim(train)
test <- data[dt,]
dim(test)

unique(train$Reason)
unique(test$Reason)

x_train = train %>% select(-X, -Body, -Title, -Tags, -AcceptedAnswerId, -CreationDate, -ClosedDate, -DeletionDate, -Reason) %>% as.matrix()
y_train = train$Reason

dtrain = xgb.DMatrix(x_train, label = y_train)
model = xgb.train(data = dtrain, nround = 150, max_depth = 7, eta = 0.1, nthread = 10, subsample = 0.9)
 
xgb.importance(feature_names = colnames(x_train), model) %>% xgb.plot.importance()
 
X_test = test %>% select(-X, -Body, -Title, -Tags, -AcceptedAnswerId, -CreationDate, -ClosedDate, -DeletionDate, -Reason) %>% as.matrix()
y_test = test$Reason
preds = predict(model, X_test)
preds = expm1(preds)
solution = data.frame(Reason = preds)
write.csv(solution, "solution.csv", row.names = FALSE)