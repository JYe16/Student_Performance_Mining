# Project 2

# Importing the dataset
dataset = read.csv('200-StudentsPerformance.csv')

# Encoding catagorical data
# c means a vector in R
dataset$gender = factor(dataset$gender,
                        levels = c('male', 'female'),
                        labels = c(1, 2))
dataset$lunch = factor(dataset$lunch,
                       levels = c('standard', 'free/reduced'),
                       labels = c(0, 1))
dataset$test.preparation.course = factor(dataset$test.preparation.course,
                                         levels = c('none', 'completed'),
                                         labels = c(0, 1))
dataset$race.ethnicity = factor(dataset$race.ethnicity,
                                levels = c('group A', 'group B', 'group C', 'group D', 'group E'),
                                labels = c(1, 2, 3, 4, 5))

# Splitting the dataset into the training set and test set
library(caTools)
set.seed(123)      # used for practice
split = sample.split(dataset$test.preparation.course, SplitRatio = 0.75) # always dependent variable
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
dataset[,6:8] = scale(dataset[,6:8])
training_set[,6:8] = scale(training_set[,6:8]) # R index starts with 1, doesnt have incl/excl
test_set[,6:8] = scale(test_set[,6:8]) # R index starts with 1, doesnt have incl/excl


# Fitting Logistic Regression to the Training Set
classifier = glm(formula = test.preparation.course ~ reading.score + writing.score,
                 family = binomial,
                 data = training_set)
# Predicting the Test set results
prob_pred = predict(classifier, type = "response", newdata = training_set[, 7:8]) # only need independent vars
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making confusion matrix for lunch type
cm = table(training_set[,4], y_pred)
# CM for logistic regression
#
# y_pred
#     0   1
# 0   353 133
# 1   219 45
#
# prediction %: 39.8%
# original %: 35.2%
# BAD RESULT

# Making confusion matrix for gender
cm = table(training_set[,1], y_pred)
# CM for logistic regression
#
# y_pred
#     0   1
# 0   315 47
# 1   257 131
#
# prediction %: 30.4%
# original %: 35.2%
# BAD RESULT

# Making confusion matrix for test prep
cm = table(training_set[,5], y_pred)
# CM for logistic regression
#
# y_pred
#     0   1
# 0   411 71
# 1   161 107
#
# prediction %: 51.8%
# original %: 48.2%
# BAD RESULT





# Visualizing the Training set results
set = training_set
X1 = seq(min(set[, 7]) - 1, max(set[, 7]) + 1, by = 0.01)
X2 = seq(min(set[, 8]) - 1, max(set[, 8]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2) # for the background,
colnames(grid_set) = c('reading.score', 'writing.score') # need names for the model
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, 6:7],
     main = 'Logistic Regression (Training Set)',
     xlab = 'Reading Score', ylab = 'Writing Score',
     xlim = range(X1), ylim = range(X2))
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set[, 6:7], pch = 21, bg = ifelse(set[4] == 1, 'green4', 'red3'))


########## Kernel-SVM ##########
# Fitting SVM
# install e1071 through terminal
library(e1071)
#Kernel SVM using gender as predictor
classifier = svm(formula = gender ~ math.score + reading.score,
                 data = dataset,
                 type = 'C-classification',
                 kernel = "radial")
#85 89 -> 174
# Predicting the Test set results
y_pred = predict(classifier, newdata = dataset)

# Making confusion matrix
cm = table(dataset[,1], y_pred)
# 82.6% correct

#Kernel SVM using Lunch as Predictor
classifier = svm(formula = lunch ~ math.score + reading.score ,
                 data = dataset,
                 type = 'C-classification',
                 kernel = "linear",
                 degree = 3)
#42 246 -> 288
# Predicting the Test set results
y_pred = predict(classifier, newdata = dataset)

# Making confusion matrix
cm = table(dataset[,4], y_pred)
# 71.2% correct

#Kernel SVM using Test Prep as predictor
classifier = svm(formula = test.preparation.course ~ math.score + reading.score,
                 data = dataset,
                 type = 'C-classification',
                 kernel = "radial",
                 degree = 3)
#24 318 -> 342

# Predicting the Test set results
y_pred = predict(classifier, newdata = dataset)

# Making confusion matrix
cm = table(dataset[,5], y_pred)
# 65.8% correct


########## Naive Bayes ##########
library(e1071)

# Gender as predictor
classifier = naiveBayes(formula = gender ~ reading.score + writing.score + math.score,
                        data = dataset)

# Predicting the Test set results
y_pred = predict(classifier, newdata = dataset)

# Making confusion matrix
cm = table(dataset[,1], y_pred)
# male = 1, female = 2
#
#     1   2
# 1  327 155
# 2  156 362
#
# When it was actually a male, 155 times the model incorrectly classifies the scores as female
# When it was actually a female, 156 times the model incorrectly classifies the scores as male
# 68.9% correct classifications, this is better than classifying everyone as male (48.2% correct)



# Lunch as predictor
classifier = naiveBayes(formula = lunch ~ reading.score + writing.score + math.score,
                        data = dataset)

# Predicting the Test set results
y_pred = predict(classifier, newdata = dataset)

# Making confusion matrix
cm = table(dataset[,4], y_pred)
# standard = 0, reduced = 1
#
#     0   1
# 0  511 134
# 1  195 160
#
# When it was actually a standard lunch, 134 times the model incorrectly classifies the scores as reduced lunch
# When it was actually a reduced lunch, 195 times the model incorrectly classifies the scores as standard lunch
# 67.1% correct classifications, only slightly better than always classifying everyone as standard lunch (64.5%)


# Prep as predictor
classifier = naiveBayes(formula = test.preparation.course ~ reading.score + writing.score + math.score,
                        data = dataset)

# Predicting the Test set results
y_pred = predict(classifier, newdata = dataset)

# Making confusion matrix
cm = table(dataset[,5], y_pred)
# none = 0, completed = 1
#
#     0   1
# 0  447 195
# 1  165 193
#
# When it was actually a no prep, 195 times the model incorrectly classifies the scores as completed prep
# When it was actually a completed prep, 165 times the model incorrectly classifies the scores as no prep
# 64.0% correct classifications, slightly WORSE than always classifying everyone as no prep (64.2%)

######### Models on a smaller data set ##########

# Visualizing the Training set results
set = dataset
X1 = seq(min(set[6]) - 1, max(set[6]) + 1, by = 0.01)
X2 = seq(min(set[7]) - 1, max(set[7]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('math.score', 'reading.score')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = predict(classifier, newdata = grid_set)
plot(set[6:7],
     main = 'SVM',
     xlab = 'Math Score', ylab = 'Reading Score',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(y_grid, length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set[6:7], pch = 21, bg = ifelse(set[1] == 1, 'green4', 'red3'))



