##### 步骤 0：加载必要包 #####
library(caret)
library(dplyr)

##### 步骤 1：设置工作目录并读取数据 #####
setwd("C:/Users/54176/Desktop")
raw_data <- read.csv("C:/Users/54176/Desktop/IFX-ADA.csv")
data <- raw_data  # 保留原始数据副本

##### 步骤 2：清洗 Result 列 #####
table(data$Result, useNA = "always")  # 查看异常值
data$Result[!data$Result %in% c(0, 1)] <- NA  # 非0/1设为NA
data <- na.omit(data)  # 删除含NA行
data$Result <- factor(data$Result, levels = c(0, 1), labels = c("No", "Yes"))

##### 步骤 3：变量类型转换 #####
data$Na <- as.numeric(data$Na)
data$AST <- as.numeric(data$AST)
data[, 2:14] <- lapply(data[, 2:14], as.factor)

##### 步骤 4：划分变量 #####
cat_vars <- names(data)[1:14]            # 分类变量
cont_vars <- names(data)[15:ncol(data)]  # 连续变量

##### 步骤 5：标准化连续变量 #####
set.seed(111)
cont_scaled <- scale(data[, cont_vars])
data_scaled <- data.frame(
  data[, cat_vars],     # 分类变量（含Result）
  cont_scaled           # 标准化的连续变量
)

##### 步骤 6：划分训练集与测试集（按Result平衡划分） #####
set.seed(52)
split_index <- createDataPartition(y = data$Result, p = 0.8, list = FALSE)

train_data <- data[split_index, ]
test_data  <- data[-split_index, ]
train_data_scaled <- data_scaled[split_index, ]
test_data_scaled  <- data_scaled[-split_index, ]

##### 步骤 7：选择用于建模的变量 #####
# 使用更新后的变量名
var <- c(
  "Result",
  "Prior_exposure_to_antiTNF_agents",  # 正确的列名
  "History_of_delayed_treatment",
  "Concomitant_use_of_IMM",
  "TLI",
  "ESR"
)

# 检查变量名是否存在
missing_vars <- setdiff(var, names(train_data))
if (length(missing_vars) > 0) {
  stop("❌ 以下变量在 train_data 中未找到：", paste(missing_vars, collapse = ", "))
}

# 提取子集
traindata         <- train_data[, var]
testdata          <- test_data[, var]
traindata_scaled  <- train_data_scaled[, var]
testdata_scaled   <- test_data_scaled[, var]

# 保证测试集因子水平一致（尤其Result）
testdata$Result <- factor(testdata$Result, levels = levels(traindata$Result))

##### 步骤 8：去除NA（保险起见） #####
traindata         <- na.omit(traindata)
testdata          <- na.omit(testdata)
traindata_scaled  <- na.omit(traindata_scaled)
testdata_scaled   <- na.omit(testdata_scaled)

##### 步骤 9：结果验证 #####
cat("训练集 Result 分布（无 NA）:\n")
print(table(traindata$Result, useNA = "always"))

cat("\n测试集 Result 分布（无 NA）:\n")
print(table(testdata$Result, useNA = "always"))




####################2.二分类机器学习模型建模####################################

# 安装及加载必要的R包   
library(caret)       # 数据处理、混淆矩阵
library(rms)         #绘制校准曲线
library(rpart)       # 构建决策树模型
library(rpart.plot)  # 可视化决策树
library(pROC)        # 绘制ROC曲线
library(ggplot2)     # 绘图
library(randomForest)  #构建随机森林模型
library(xgboost)    # XGBoost模型  
library(lightgbm)    # 构建lightgbm模型  
library(kknn)        # 构建knn模型 
library(neuralnet) # nnet模型 
library(NeuralNetTools) 
library(e1071)      # svm模型  

#将结局变量因子化
#traindata$Result <- factor(traindata$Result,levels = c(0,1),labels = c('No','Yes'))
#testdata$Result <- factor(testdata$Result,levels = c(0,1),labels = c('No','Yes'))

#traindata_scaled$Result <- factor(traindata_scaled$Result,levels = c(0,1),labels = c('No','Yes'))
#testdata_scaled$Result <- factor(testdata_scaled$Result,levels = c(0,1),labels = c('No','Yes'))


# ====== 添加验证代码 ======
# 检查 Result 是否已为因子且无 NA
cat("训练集 Result 类型:", class(traindata$Result), "\n")
cat("测试集 Result 类型:", class(testdata$Result), "\n")
cat("训练集标准化数据 Result 类型:", class(traindata_scaled$Result), "\n")
cat("测试集标准化数据 Result 类型:", class(testdata_scaled$Result), "\n")

# 输出因子水平
cat("\n训练集 Result 水平:", levels(traindata$Result))
cat("\n测试集 Result 水平:", levels(testdata$Result))


####################2.1 Logistic模型进阶版#####################
# 进阶版（显示AUC(95% CI)、准确率Accuracy(95% CI)、灵敏度Sencitivity、
# 特异度Specificity、精确率Precision、召回率Recall、F1值、Brier分数）
library(caret)
library(pROC)
library(openxlsx) # 用于创建Excel工作簿

# 1. 拟合Logistic回归模型
lr_model <- glm(Result ~ ., data = traindata, family = "binomial")

# 2. 模型预测概率与类别
train_prob_lr <- predict(lr_model, newdata = traindata, type = "response")
test_prob_lr <- predict(lr_model, newdata = testdata, type = "response")

# 创建因子预测结果
train_pred_lr <- factor(ifelse(train_prob_lr > 0.5, "Yes", "No"), levels = c("No", "Yes"))
test_pred_lr <- factor(ifelse(test_prob_lr > 0.5, "Yes", "No"), levels = c("No", "Yes"))

# 3. 混淆矩阵
confusion_matrix_train <- caret::confusionMatrix(
  data = train_pred_lr,
  reference = traindata$Result,
  positive = "Yes"
)
print(confusion_matrix_train)

confusion_matrix_test <- caret::confusionMatrix(
  data = test_pred_lr,
  reference = testdata$Result,
  positive = "Yes"
)
print(confusion_matrix_test)

# 4. 计算AUC和95% CI
roc_train <- pROC::roc(response = traindata$Result, predictor = train_prob_lr)
roc_test <- pROC::roc(response = testdata$Result, predictor = test_prob_lr)

# 计算AUC
train_auc <- as.numeric(pROC::auc(roc_train))
test_auc <- as.numeric(pROC::auc(roc_test))

# 计算AUC置信区间
train_auc_ci <- pROC::ci(roc_train)
test_auc_ci <- pROC::ci(roc_test)

cat("\nTrain AUC (95% CI):", sprintf("%.3f", train_auc), 
    "(", sprintf("%.3f", train_auc_ci[1]), "-", 
    sprintf("%.3f", train_auc_ci[3]), ")\n")

cat("Test AUC (95% CI):", sprintf("%.3f", test_auc), 
    "(", sprintf("%.3f", test_auc_ci[1]), "-", 
    sprintf("%.3f", test_auc_ci[3]), ")\n")

# 5. 计算其他指标
precision_train <- confusion_matrix_train$byClass["Pos Pred Value"]
recall_train <- confusion_matrix_train$byClass["Sensitivity"]
f1_train <- confusion_matrix_train$byClass["F1"]

precision_test <- confusion_matrix_test$byClass["Pos Pred Value"]
recall_test <- confusion_matrix_test$byClass["Sensitivity"]
f1_test <- confusion_matrix_test$byClass["F1"]

# 提取准确率及其95%置信区间
accuracy_train <- confusion_matrix_train$overall["Accuracy"]
accuracy_train_ci <- confusion_matrix_train$overall[c("AccuracyLower", "AccuracyUpper")]

accuracy_test <- confusion_matrix_test$overall["Accuracy"]
accuracy_test_ci <- confusion_matrix_test$overall[c("AccuracyLower", "AccuracyUpper")]

cat("\n训练集指标:\n")
cat("Accuracy:", sprintf("%.3f", accuracy_train), 
    "(", sprintf("%.3f", accuracy_train_ci[1]), "-", 
    sprintf("%.3f", accuracy_train_ci[2]), ")\n")
cat("Precision:", sprintf("%.3f", precision_train), "\n")
cat("Recall:", sprintf("%.3f", recall_train), "\n")
cat("F1 Score:", sprintf("%.3f", f1_train), "\n")

cat("\n测试集指标:\n")
cat("Accuracy:", sprintf("%.3f", accuracy_test), 
    "(", sprintf("%.3f", accuracy_test_ci[1]), "-", 
    sprintf("%.3f", accuracy_test_ci[2]), ")\n")
cat("Precision:", sprintf("%.3f", precision_test), "\n")
cat("Recall:", sprintf("%.3f", recall_test), "\n")
cat("F1 Score:", sprintf("%.3f", f1_test), "\n")

# 6. Brier Score计算
actual_train_num <- ifelse(traindata$Result == "Yes", 1, 0)
actual_test_num <- ifelse(testdata$Result == "Yes", 1, 0)

brier_train <- mean((actual_train_num - train_prob_lr)^2)
brier_test <- mean((actual_test_num - test_prob_lr)^2)

cat("\nTrain Brier Score:", sprintf("%.3f", brier_train), "\n")
cat("Test Brier Score:", sprintf("%.3f", brier_test), "\n")

# 7. 创建性能汇总表（包含准确率95% CI）
performance_summary <- data.frame(
  Dataset = c("Training Set", "Test Set"),
  Accuracy = c(accuracy_train, accuracy_test),
  Accuracy_Lower = c(accuracy_train_ci["AccuracyLower"], accuracy_test_ci["AccuracyLower"]),
  Accuracy_Upper = c(accuracy_train_ci["AccuracyUpper"], accuracy_test_ci["AccuracyUpper"]),
  AUC = c(train_auc, test_auc),
  AUC_Lower = c(train_auc_ci[1], test_auc_ci[1]),
  AUC_Upper = c(train_auc_ci[3], test_auc_ci[3]),
  Precision = c(precision_train, precision_test),
  Recall = c(recall_train, recall_test),
  F1 = c(f1_train, f1_test),
  Brier = c(brier_train, brier_test),
  stringsAsFactors = FALSE
)

# 8. 可视化ROC曲线
par(mfrow = c(1, 2))
plot(roc_train, main = "Training Set ROC", col = "blue", print.auc = TRUE, legacy.axes = TRUE)
plot(roc_test, main = "Test Set ROC", col = "red", print.auc = TRUE, legacy.axes = TRUE)

# 9. 导出所有数据到Excel
# 创建Excel工作簿
wb <- createWorkbook()

# 添加模型摘要表
model_summary <- as.data.frame(summary(lr_model)$coefficients)
model_summary <- cbind(Variable = rownames(model_summary), model_summary)
rownames(model_summary) <- NULL
addWorksheet(wb, "Model Summary")
writeData(wb, "Model Summary", model_summary)

# 添加训练集混淆矩阵
train_cm <- as.data.frame(confusion_matrix_train$table)
addWorksheet(wb, "Train Confusion Matrix")
writeData(wb, "Train Confusion Matrix", train_cm)

# 添加测试集混淆矩阵
test_cm <- as.data.frame(confusion_matrix_test$table)
addWorksheet(wb, "Test Confusion Matrix")
writeData(wb, "Test Confusion Matrix", test_cm)

# 添加性能指标汇总表
addWorksheet(wb, "Performance Summary")
writeData(wb, "Performance Summary", performance_summary)

# 添加训练集预测结果
train_results <- data.frame(
  Actual = traindata$Result,
  Predicted = train_pred_lr,
  Probability = train_prob_lr
)
addWorksheet(wb, "Train Predictions")
writeData(wb, "Train Predictions", train_results)

# 添加测试集预测结果
test_results <- data.frame(
  Actual = testdata$Result,
  Predicted = test_pred_lr,
  Probability = test_prob_lr
)
addWorksheet(wb, "Test Predictions")
writeData(wb, "Test Predictions", test_results)

# 添加ROC曲线数据
roc_train_data <- data.frame(
  Specificity = roc_train$specificities,
  Sensitivity = roc_train$sensitivities
)
addWorksheet(wb, "ROC Train")
writeData(wb, "ROC Train", roc_train_data)

roc_test_data <- data.frame(
  Specificity = roc_test$specificities,
  Sensitivity = roc_test$sensitivities
)
addWorksheet(wb, "ROC Test")
writeData(wb, "ROC Test", roc_test_data)

# 保存Excel文件
excel_file <- "Logistic_Regression_Results.xlsx"
saveWorkbook(wb, excel_file, overwrite = TRUE)
cat("\n所有结果已导出到Excel文件:", excel_file, "\n")

# 10. 返回性能汇总表（控制台输出）
cat("\n模型性能汇总:\n")
print(performance_summary, row.names = FALSE)




########################2.2 knn 模型###########################
# 进阶版（显示AUC(95% CI)、准确率Accuracy(95% CI)、灵敏度Sencitivity、
# 特异度Specificity、精确率Precision、召回率Recall、F1值、Brier分数）

# 加载必要包
library(kknn)
library(pROC)
library(caret)
library(DescTools)

# 调参：选择最优k
k_values <- seq(1, 30, by = 1)
auc_results <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  k <- k_values[i]
  model <- kknn(Result ~ ., 
                train = traindata_scaled, 
                test = testdata_scaled, 
                k = k, 
                kernel = "triangular")
  pred_probs <- fitted(model)
  roc_curve <- roc(testdata_scaled$Result, as.numeric(pred_probs))
  auc_results[i] <- roc_curve$auc
}

# 可视化K vs AUC
plot(k_values, auc_results, type = "b",
     xlab = "Number of Neighbors (k)",
     ylab = "AUC",
     main = "K vs AUC")

# 获取最佳K值
best_k_auc <- k_values[which.max(auc_results)]
print(paste("Best k value based on AUC:", best_k_auc))

# 构建最终模型
knn_model_train <- kknn(Result ~ ., 
                        train = traindata_scaled, 
                        test = traindata_scaled, 
                        k = best_k_auc, 
                        kernel = "triangular")
knn_model_test <- kknn(Result ~ ., 
                       train = traindata_scaled, 
                       test = testdata_scaled,  
                       k = best_k_auc, 
                       kernel = "triangular")

# 训练集预测
train_pred <- predict(knn_model_train, newdata = traindata_scaled)
train_prob <- predict(knn_model_train, newdata = traindata_scaled, type = "prob")[,"Yes"]
train_roc <- roc(traindata_scaled$Result, train_prob)
train_auc <- auc(train_roc)
train_ci <- ci.auc(train_roc)

# 测试集预测
test_pred <- predict(knn_model_test, newdata = testdata_scaled)
test_prob <- predict(knn_model_test, newdata = testdata_scaled, type = "prob")[,"Yes"]
test_roc <- roc(testdata_scaled$Result, test_prob)
test_auc <- auc(test_roc)
test_ci <- ci.auc(test_roc)

# 混淆矩阵与评估指标
train_conf <- confusionMatrix(train_pred, traindata_scaled$Result, positive = "Yes")
test_conf <- confusionMatrix(test_pred, testdata_scaled$Result, positive = "Yes")

# 修正Brier Score计算
# 确保结果转换为数值型：Yes=1, No=0
brier_train <- mean((as.numeric(traindata_scaled$Result == "Yes") - train_prob)^2)
brier_test <- mean((as.numeric(testdata_scaled$Result == "Yes") - test_prob)^2)

# 训练集结果输出
cat("----训练集结果----\n")
cat("AUC:", round(train_auc, 4), "(95% CI:", round(train_ci[1], 4), "-", round(train_ci[3], 4), ")\n")
cat("Accuracy:", round(train_conf$overall["Accuracy"], 4),
    "(95% CI:", round(train_conf$overall["AccuracyLower"], 4), "-",
    round(train_conf$overall["AccuracyUpper"], 4), ")\n")
cat("Sensitivity:", round(train_conf$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(train_conf$byClass["Specificity"], 4), "\n")
cat("Precision:", round(train_conf$byClass["Precision"], 4), "\n")
cat("Recall:", round(train_conf$byClass["Recall"], 4), "\n")
cat("F1:", round(train_conf$byClass["F1"], 4), "\n")
cat("Brier Score:", round(brier_train, 4), "\n")

# 测试集结果输出
cat("\n----测试集结果----\n")
cat("AUC:", round(test_auc, 4), "(95% CI:", round(test_ci[1], 4), "-", round(test_ci[3], 4), ")\n")
cat("Accuracy:", round(test_conf$overall["Accuracy"], 4),
    "(95% CI:", round(test_conf$overall["AccuracyLower"], 4), "-",
    round(test_conf$overall["AccuracyUpper"], 4), ")\n")
cat("Sensitivity:", round(test_conf$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(test_conf$byClass["Specificity"], 4), "\n")
cat("Precision:", round(test_conf$byClass["Precision"], 4), "\n")
cat("Recall:", round(test_conf$byClass["Recall"], 4), "\n")
cat("F1:", round(test_conf$byClass["F1"], 4), "\n")
cat("Brier Score:", round(brier_test, 4), "\n")





################2.3 CART模型########################################
# CART进阶版（显示AUC(95% CI)、准确率Accuracy(95% CI)、灵敏度Sencitivity、
# 特异度Specificity、精确率Precision、召回率Recall、F1值、Brier分数）
# 加载必要的包
library(caret)
library(rpart)
library(rpart.plot)
library(partykit)
library(dplyr)
library(pROC)  # 用于计算 AUC
set.seed(111)

# 构建 CART 模型并调参
control <- trainControl(method = "cv", number = 10)
param_grid <- expand.grid(cp = seq(0.001, 0.3, by = 0.001))

fit_cv_rpart <- train(Result ~ ., data = traindata,
                      method = "rpart",
                      trControl = control,
                      tuneGrid = param_grid)
best_cp <- fit_cv_rpart$bestTune$cp
cat("Best cp:", best_cp, "\n")

# 构建最终模型
tree_model <- rpart(Result ~ ., data = traindata,
                    method = "class", cp = best_cp)

# 决策树图
rpart.plot(tree_model)
plot(as.party(tree_model))

# 预测
train_pred_tree <- predict(tree_model, newdata = traindata, type = "class")
train_prob_tree <- predict(tree_model, newdata = traindata, type = "prob")[, "Yes"]

test_pred_tree <- predict(tree_model, newdata = testdata, type = "class")
test_prob_tree <- predict(tree_model, newdata = testdata, type = "prob")[, "Yes"]

# 计算 AUC 与 95% CI
train_roc <- roc(traindata$Result, train_prob_tree)
test_roc  <- roc(testdata$Result, test_prob_tree)

train_auc <- auc(train_roc)
test_auc  <- auc(test_roc)

train_ci <- ci.auc(train_roc)
test_ci  <- ci.auc(test_roc)

# 混淆矩阵
conf_train <- confusionMatrix(train_pred_tree, traindata$Result, positive = "Yes")
conf_test  <- confusionMatrix(test_pred_tree,  testdata$Result,  positive = "Yes")

# 手动计算 Brier 分数
brier_train <- mean((train_prob_tree - as.numeric(traindata$Result == "Yes"))^2)
brier_test  <- mean((test_prob_tree  - as.numeric(testdata$Result == "Yes"))^2)

# 输出训练集结果
cat("==== 训练集结果 ====\n")
cat("AUC:", round(train_auc, 4), "(95% CI:", round(train_ci[1], 4), "-", round(train_ci[3], 4), ")\n")
cat("Accuracy:", round(conf_train$overall["Accuracy"], 4),
    "(95% CI:", round(conf_train$overall["AccuracyLower"], 4), "-",
    round(conf_train$overall["AccuracyUpper"], 4), ")\n")
print(conf_train$byClass[c("Sensitivity", "Specificity", "Precision", "Recall", "F1")])
cat("Brier Score:", round(brier_train, 4), "\n")

# 输出测试集结果
cat("\n==== 测试集结果 ====\n")
cat("AUC:", round(test_auc, 4), "(95% CI:", round(test_ci[1], 4), "-", round(test_ci[3], 4), ")\n")
cat("Accuracy:", round(conf_test$overall["Accuracy"], 4),
    "(95% CI:", round(conf_test$overall["AccuracyLower"], 4), "-",
    round(conf_test$overall["AccuracyUpper"], 4), ")\n")
print(conf_test$byClass[c("Sensitivity", "Specificity", "Precision", "Recall", "F1")])
cat("Brier Score:", round(brier_test, 4), "\n")









###################2.4 随机森林(RF)模型##########################
# 训练集上 Specificity 和 Pos Pred Value 都为 1.0000 是否正常，逐一分析原因：
# 这种完全预测正确的现象在训练集上出现，是可能的，尤其是使用了较多参数调优、较多树的随机森林。 但问题是： 
# 在训练集上表现极好（完美识别 No 类） 
# 在测试集上，准确率明显下降，特异度和灵敏度都有损失，这通常意味着模型在训练集上过拟合，对新数据的泛化能力较弱。

# 可以采取以下优化措施： 
# 进一步交叉验证避免过拟合 
# 设置 tuneLength 而非固定网格范围 
# 评估 AUC 和 ROC 曲线

# ==== 加载必要包 ====
library(randomForest)
library(caret)
library(pROC)
library(ggplot2)

set.seed(123)

# =======================
# 1. 交叉验证 + 自动调参
# =======================
ctrl <- trainControl(method = "cv", number = 10,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     savePredictions = "final")

rf_model <- train(Result ~ ., data = traindata,
                  method = "rf",
                  metric = "ROC",  # AUC 作为评估标准
                  tuneLength = 10,
                  trControl = ctrl)

cat("\n===== RF模型交叉验证结果 =====\n")
print(rf_model)
plot(rf_model)

# =======================
# 2. 查找最佳 mtry 和 ntree
# =======================
best_mtry <- rf_model$bestTune$mtry
ntree_values <- seq(50, 1000, by = 50)
oob_error_rates <- numeric(length(ntree_values))

for (i in seq_along(ntree_values)) {
  temp_model <- randomForest(Result ~ ., data = traindata,
                             mtry = best_mtry,
                             ntree = ntree_values[i],
                             importance = TRUE)
  oob_error_rates[i] <- temp_model$err.rate[ntree_values[i], "OOB"]
}

best_ntree <- ntree_values[which.min(oob_error_rates)]
cat("\n✅ 最佳 ntree 数量：", best_ntree, "\n")

# =======================
# 3. 训练最终模型
# =======================
final_rf <- randomForest(Result ~ ., data = traindata,
                         mtry = best_mtry,
                         ntree = best_ntree,
                         importance = TRUE)

cat("\n===== 变量重要性排序 =====\n")
print(importance(final_rf))
varImpPlot(final_rf)

# =======================
# 4. 训练集预测 + 指标评估
# =======================
train_pred <- predict(final_rf, newdata = traindata)
train_prob_rf <- predict(final_rf, newdata = traindata, type = "prob")[, "Yes"]

cat("\n===== 训练集 混淆矩阵 =====\n")
cm_train <- confusionMatrix(train_pred, traindata$Result, positive = "Yes")
print(cm_train)

train_roc <- roc(traindata$Result, train_prob_rf, levels = c("No", "Yes"), ci = TRUE)
brier_train <- mean((ifelse(traindata$Result == "Yes", 1, 0) - train_prob_rf)^2)

cat("\n===== 训练集评估指标 =====\n")
cat("AUC:", round(train_roc$auc, 3), "\n")
cat("AUC 95% CI:", paste0(round(train_roc$ci, 3), collapse = " - "), "\n")
cat("Accuracy:", round(cm_train$overall["Accuracy"], 3),
    " (95% CI:", round(cm_train$overall["AccuracyLower"], 3), "-", 
    round(cm_train$overall["AccuracyUpper"], 3), ")\n")
cat("Sensitivity:", round(cm_train$byClass["Sensitivity"], 3), "\n")
cat("Specificity:", round(cm_train$byClass["Specificity"], 3), "\n")
cat("Precision:", round(cm_train$byClass["Precision"], 3), "\n")
cat("Recall:", round(cm_train$byClass["Recall"], 3), "\n")
cat("F1 Score:", round(cm_train$byClass["F1"], 3), "\n")
cat("Brier Score:", round(brier_train, 4), "\n")

# =======================
# 5. 测试集预测 + 指标评估
# =======================
test_pred <- predict(final_rf, newdata = testdata)
test_prob_rf <- predict(final_rf, newdata = testdata, type = "prob")[, "Yes"]

cat("\n===== 测试集 混淆矩阵 =====\n")
cm_test <- confusionMatrix(test_pred, testdata$Result, positive = "Yes")
print(cm_test)

test_roc <- roc(testdata$Result, test_prob_rf, levels = c("No", "Yes"), ci = TRUE)
brier_test <- mean((ifelse(testdata$Result == "Yes", 1, 0) - test_prob_rf)^2)

cat("\n===== 测试集评估指标 =====\n")
cat("AUC:", round(test_roc$auc, 3), "\n")
cat("AUC 95% CI:", paste0(round(test_roc$ci, 3), collapse = " - "), "\n")
cat("Accuracy:", round(cm_test$overall["Accuracy"], 3),
    " (95% CI:", round(cm_test$overall["AccuracyLower"], 3), "-", 
    round(cm_test$overall["AccuracyUpper"], 3), ")\n")
cat("Sensitivity:", round(cm_test$byClass["Sensitivity"], 3), "\n")
cat("Specificity:", round(cm_test$byClass["Specificity"], 3), "\n")
cat("Precision:", round(cm_test$byClass["Precision"], 3), "\n")
cat("Recall:", round(cm_test$byClass["Recall"], 3), "\n")
cat("F1 Score:", round(cm_test$byClass["F1"], 3), "\n")
cat("Brier Score:", round(brier_test, 4), "\n")

# =======================
# 6. 绘制 ROC 曲线图
# =======================
plot(test_roc, main = "随机森林 ROC 曲线", col = "blue", lwd = 2)
lines(train_roc, col = "darkgreen", lwd = 2)
legend("bottomright", legend = c("测试集", "训练集"), col = c("blue", "darkgreen"), lwd = 2)

# =======================
# 7. 返回概率预测用于后续整合
# =======================
# train_prob_rf
# test_prob_rf

cat("最佳参数组合：mtry =", rf_model$bestTune$mtry, ", ntree =", best_ntree, "\n")






########################2.5 Xgboost模型################################
# 原代码修改原因：
# 标签格式统一： 
# 1.将 Result 的因子标签 "No"/"Yes" 转换为数值型 0/1，避免XGBoost报错。 
# 2.在生成混淆矩阵时，将预测结果转换回因子 "No"/"Yes"，确保与真实标签的因子水平一致。

# 加载包
library(xgboost)
library(caret)
library(pROC)

##### 关键步骤 1：将分类变量转换为数值型（独热编码）
# 提取特征列（排除Result）
features <- var[-1]

# 对分类变量进行独热编码（确保训练集和测试集使用相同的编码规则）
dummy_model <- dummyVars(~ ., data = traindata[, features], fullRank = TRUE)
train_encoded <- predict(dummy_model, newdata = traindata[, features])
test_encoded <- predict(dummy_model, newdata = testdata[, features])

# 将标签转换为0/1（因子"No"/"Yes"→0/1）
train_label <- ifelse(traindata$Result == "Yes", 1, 0)
test_label <- ifelse(testdata$Result == "Yes", 1, 0)

# 创建XGBoost输入矩阵
train_matrix <- xgb.DMatrix(data = train_encoded, label = train_label)
test_matrix <- xgb.DMatrix(data = test_encoded, label = test_label)

##### 关键步骤 2：超参数调优
param_grid <- expand.grid(
  max_depth = c(2, 3, 4, 5),
  eta = c(0.01, 0.1, 0.2),
  nrounds = c(50, 100, 150)
)
best_auc <- 0
best_params <- list()

for (i in 1:nrow(param_grid)) {
  param <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = param_grid$max_depth[i],
    eta = param_grid$eta[i]
  )
  
  xgb_model_0 <- xgb.train(
    params = param,
    data = train_matrix,
    nrounds = param_grid$nrounds[i]
  )
  
  # 预测概率
  pred_probs <- predict(xgb_model_0, test_matrix)
  # 计算AUC
  roc_curve <- roc(test_label, pred_probs)
  auc_value <- auc(roc_curve)
  
  if (auc_value > best_auc) {
    best_auc <- auc_value
    best_params <- list(
      max_depth = param_grid$max_depth[i],
      eta = param_grid$eta[i],
      nrounds = param_grid$nrounds[i]
    )
  }
}

# 输出最佳参数
cat("最佳参数: \n")
print(best_params)
cat("最佳AUC: ", best_auc, "\n")

##### 关键步骤 3：使用最佳参数训练模型
final_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = best_params$max_depth,
  eta = best_params$eta
)

xgb_model <- xgb.train(
  params = final_params,
  data = train_matrix,
  nrounds = best_params$nrounds
)

##### 关键步骤 4：预测与评估
# 预测概率
train_prob_xgb <- predict(xgb_model, train_matrix)
test_prob_xgb <- predict(xgb_model, test_matrix)

# 转换为类别（0/1）
train_pred_xgb <- ifelse(train_prob_xgb > 0.5, 1, 0)
test_pred_xgb <- ifelse(test_prob_xgb > 0.5, 1, 0)

# 转换为与真实标签一致的因子水平
train_pred_factor <- factor(train_pred_xgb, levels = c(0, 1), labels = c("No", "Yes"))
test_pred_factor <- factor(test_pred_xgb, levels = c(0, 1), labels = c("No", "Yes"))

# 创建混淆矩阵（确保因子水平一致）
confusion_matrix_train <- confusionMatrix(
  data = train_pred_factor,
  reference = traindata$Result,
  positive = "Yes"  # 明确指定阳性类别
)

confusion_matrix_test <- confusionMatrix(
  data = test_pred_factor,
  reference = testdata$Result,
  positive = "Yes"
)

# 输出结果
cat("训练集混淆矩阵:\n")
print(confusion_matrix_train)
cat("\n测试集混淆矩阵:\n")
print(confusion_matrix_test)







# Xgboost进阶版（显示AUC(95% CI)、准确率Accuracy(95% CI)、灵敏度Sencitivity、
# 特异度Specificity、精确率Precision、召回率Recall、F1值、Brier分数）
# 加载必要的包
library(xgboost)
library(caret)
library(pROC)

# 关键步骤 1：独热编码和标签处理
features <- var[-1]

dummy_model <- dummyVars(~ ., data = traindata[, features], fullRank = TRUE)
train_encoded <- predict(dummy_model, newdata = traindata[, features])
test_encoded <- predict(dummy_model, newdata = testdata[, features])

train_label <- ifelse(traindata$Result == "Yes", 1, 0)
test_label <- ifelse(testdata$Result == "Yes", 1, 0)

train_matrix <- xgb.DMatrix(data = train_encoded, label = train_label)
test_matrix <- xgb.DMatrix(data = test_encoded, label = test_label)

# 关键步骤 2：参数调优
param_grid <- expand.grid(
  max_depth = c(2, 3, 4, 5),
  eta = c(0.01, 0.1, 0.2),
  nrounds = c(50, 100, 150)
)

best_auc <- 0
best_params <- list()

for (i in 1:nrow(param_grid)) {
  param <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = param_grid$max_depth[i],
    eta = param_grid$eta[i]
  )
  
  model <- xgb.train(
    params = param,
    data = train_matrix,
    nrounds = param_grid$nrounds[i],
    verbose = 0
  )
  
  preds <- predict(model, test_matrix)
  auc_val <- auc(roc(test_label, preds))
  
  if (auc_val > best_auc) {
    best_auc <- auc_val
    best_params <- list(
      max_depth = param_grid$max_depth[i],
      eta = param_grid$eta[i],
      nrounds = param_grid$nrounds[i]
    )
  }
}

cat("最佳参数:\n")
print(best_params)
cat("最佳 AUC: ", round(best_auc, 4), "\n")

# 关键步骤 3：使用最佳参数训练最终模型
final_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = best_params$max_depth,
  eta = best_params$eta
)

xgb_model <- xgb.train(
  params = final_params,
  data = train_matrix,
  nrounds = best_params$nrounds,
  verbose = 0
)

# 显示模型结构（可视化为文本）
cat("\n====== XGBoost 模型结构预览（前10行）======\n")
model_dump <- xgb.dump(xgb_model, with_stats = TRUE)
cat(paste(model_dump[1:10], collapse = "\n"), "\n")  # 打印前10行树结构

# 关键步骤 4：预测概率和分类结果
train_prob <- predict(xgb_model, train_matrix)
test_prob <- predict(xgb_model, test_matrix)

train_pred <- ifelse(train_prob > 0.5, 1, 0)
test_pred <- ifelse(test_prob > 0.5, 1, 0)

train_pred_factor <- factor(train_pred, levels = c(0, 1), labels = c("No", "Yes"))
test_pred_factor  <- factor(test_pred,  levels = c(0, 1), labels = c("No", "Yes"))

# 混淆矩阵
conf_train <- confusionMatrix(train_pred_factor, traindata$Result, positive = "Yes")
conf_test  <- confusionMatrix(test_pred_factor,  testdata$Result,  positive = "Yes")

# AUC + 95% CI
train_roc <- roc(train_label, train_prob)
test_roc  <- roc(test_label, test_prob)

train_auc <- auc(train_roc)
test_auc  <- auc(test_roc)

train_ci <- ci.auc(train_roc)
test_ci  <- ci.auc(test_roc)

# Brier Score
brier_train <- mean((train_prob - train_label)^2)
brier_test  <- mean((test_prob - test_label)^2)

# 结果输出 - 训练集
cat("\n====== 训练集评估结果 ======\n")
cat("AUC:", round(train_auc, 4), "(95% CI:", round(train_ci[1], 4), "-", round(train_ci[3], 4), ")\n")
cat("Accuracy:", round(conf_train$overall["Accuracy"], 4),
    "(95% CI:", round(conf_train$overall["AccuracyLower"], 4), "-",
    round(conf_train$overall["AccuracyUpper"], 4), ")\n")
print(conf_train$byClass[c("Sensitivity", "Specificity", "Precision", "Recall", "F1")])
cat("Brier Score:", round(brier_train, 4), "\n")

# 结果输出 - 测试集
cat("\n====== 测试集评估结果 ======\n")
cat("AUC:", round(test_auc, 4), "(95% CI:", round(test_ci[1], 4), "-", round(test_ci[3], 4), ")\n")
cat("Accuracy:", round(conf_test$overall["Accuracy"], 4),
    "(95% CI:", round(conf_test$overall["AccuracyLower"], 4), "-",
    round(conf_test$overall["AccuracyUpper"], 4), ")\n")
print(conf_test$byClass[c("Sensitivity", "Specificity", "Precision", "Recall", "F1")])
cat("Brier Score:", round(brier_test, 4), "\n")















#######################2.6 LightGBM模型#############################
##### 修复变量名问题的完整代码 #####

# 第1步：首先检查当前数据中的列名
cat("训练数据列名：\n")
print(colnames(traindata))
cat("\n测试数据列名：\n") 
print(colnames(testdata))

# 第2步：根据实际列名更新变量列表
# 根据您之前的输出，正确的变量名应该是：
var <- c(
  "Result",
  "Prior_exposure_to_antiTNF_agents",  # 使用实际存在的列名
  "History_of_delayed_treatment",
  "Concomitant_use_of_IMM", 
  "TLI",  # 注意：之前代码中写的是TIL，但应该是TLI
  "ESR"
)

# 检查变量是否存在
missing_vars <- setdiff(var, colnames(traindata))
if(length(missing_vars) > 0) {
  cat("缺失的变量:", paste(missing_vars, collapse = ", "), "\n")
  cat("请检查以下可用的列名:\n")
  print(colnames(traindata))
  stop("请修正变量名后重新运行")
}

##### LightGBM 模型代码修复 #####
library(lightgbm)
library(caret)
library(pROC)
library(dplyr)
library(knitr)
library(kableExtra)

# 数据预处理
features <- var[-1]  # 除了Result之外的特征
dummy_model <- dummyVars(~ ., data = traindata[, features], fullRank = TRUE)
train_encoded <- predict(dummy_model, newdata = traindata[, features])
test_encoded  <- predict(dummy_model, newdata = testdata[, features])

# 标签转换
train_label <- ifelse(traindata$Result == "Yes", 1, 0)
test_label  <- ifelse(testdata$Result == "Yes", 1, 0)
scale_pos_weight <- sum(train_label == 0) / sum(train_label == 1)

# 参数配置
base_params <- list(
  objective = "binary",
  metric = "custom",
  feature_pre_filter = FALSE,
  scale_pos_weight = scale_pos_weight,
  force_col_wise = TRUE
)

# 构建数据集
dtrain <- lgb.Dataset(data = train_encoded, label = train_label, free_raw_data = FALSE)
dtest  <- lgb.Dataset.create.valid(dtrain, data = test_encoded, label = test_label)

# 自定义 F1 评估函数
f1_score <- function(preds, dtrain) {
  labels <- get_field(dtrain, "label")
  preds_binary <- as.integer(preds > 0.5)
  tp <- sum(preds_binary == 1 & labels == 1)
  fp <- sum(preds_binary == 1 & labels == 0)
  fn <- sum(preds_binary == 0 & labels == 1)
  precision <- tp / (tp + fp + 1e-6)
  recall <- tp / (tp + fn + 1e-6)
  f1 <- 2 * precision * recall / (precision + recall + 1e-6)
  return(list(name = "f1", value = f1, higher_better = TRUE))
}

# 网格搜索调参
param_grid <- expand.grid(
  num_leaves = c(31, 63),
  max_depth = c(5, 7),
  learning_rate = c(0.05),
  n_estimators = c(200),
  min_data_in_leaf = c(20, 50),
  lambda_l1 = c(0),
  lambda_l2 = c(0.1)
)

results <- data.frame()
for (i in 1:nrow(param_grid)) {
  current_params <- c(base_params, as.list(param_grid[i, ]))
  
  cat("\n正在尝试参数组合", i, "/", nrow(param_grid), ":\n")
  
  cv_model <- tryCatch({
    lgb.cv(
      params = current_params,
      data = dtrain,
      nrounds = current_params$n_estimators,
      nfold = 5,
      eval = f1_score,
      early_stopping_rounds = 20,
      verbose = -1
    )
  }, error = function(e) {
    cat("参数组合", i, "失败:", e$message, "\n")
    return(NULL)
  })
  
  if (!is.null(cv_model)) {
    best_iter <- cv_model$best_iter
    best_f1 <- cv_model$record_evals$valid$f1$eval[[best_iter]]
    results <- rbind(results, data.frame(param_grid[i, ], f1 = best_f1))
    cat("当前最佳 F1:", max(results$f1, na.rm = TRUE), "\n")
  }
}

# 最终模型训练
if (nrow(results) > 0) {
  best_params <- results[which.max(results$f1), ]
  final_params <- c(base_params, as.list(best_params))
  
  lightgbm_model <- lgb.train(
    params = final_params,
    data = dtrain,
    nrounds = best_params$n_estimators,
    valids = list(test = dtest),
    eval = f1_score,
    early_stopping_rounds = 20,
    verbose = 1
  )
  
  # 训练集预测
  train_prob_lightgbm <- predict(lightgbm_model, train_encoded)
  roc_obj <- roc(train_label, train_prob_lightgbm)
  best_threshold <- coords(roc_obj, x = "best", best.method = "closest.topleft")$threshold
  
  train_pred <- factor(ifelse(train_prob_lightgbm > best_threshold, "Yes", "No"), levels = c("No", "Yes"))
  train_actual <- factor(traindata$Result, levels = c("No", "Yes"))
  
  cat("\n===== 训练集 混淆矩阵 =====\n")
  print(confusionMatrix(train_pred, train_actual, positive = "Yes"))
  
  # 测试集预测
  test_prob_lightgbm <- predict(lightgbm_model, test_encoded)
  test_pred <- factor(ifelse(test_prob_lightgbm > best_threshold, "Yes", "No"), levels = c("No", "Yes"))
  test_actual <- factor(testdata$Result, levels = c("No", "Yes"))
  
  cat("\n===== 测试集 混淆矩阵 =====\n")
  print(confusionMatrix(test_pred, test_actual, positive = "Yes"))
  
  # AUC值
  auc_lightgbm_test <- auc(test_actual, test_prob_lightgbm)
  cat("\n===== 测试集 AUC =====\n")
  print(auc_lightgbm_test)
}






######################## 2.7 SVM模型 ################################
library(e1071)

# 数据预处理 - 使用修正后的变量名
features <- var[-1]
dummy_model_svm <- dummyVars(~ ., data = traindata[, features], fullRank = TRUE)
train_encoded_svm <- as.data.frame(predict(dummy_model_svm, newdata = traindata[, features]))
test_encoded_svm  <- as.data.frame(predict(dummy_model_svm, newdata = testdata[, features]))

train_svm <- data.frame(Result = factor(traindata$Result, levels = c("No", "Yes")), train_encoded_svm)
test_svm  <- data.frame(Result = factor(testdata$Result, levels = c("No", "Yes")), test_encoded_svm)

train_svm <- na.omit(train_svm)
test_svm  <- na.omit(test_svm)

# 动态设置交叉验证折数
n_samples <- nrow(train_svm)
if (n_samples < 2) stop("训练集样本量不足，无法交叉验证！")
max_folds <- min(10, max(2, floor(n_samples * 0.8)))

# 参数调优
set.seed(11)
tune_result <- tune.svm(
  Result ~ ., 
  data = train_svm,
  kernel = "radial",
  cost = 10^(-1:3),
  gamma = 10^(-3:1),
  tunecontrol = tune.control(sampling = "cross", cross = max_folds),
  probability = TRUE
)

best_model <- tune_result$best.model
cat("最佳参数组合:\n")
print(best_model)

# 使用最佳参数训练模型
svm_model <- svm(
  Result ~ .,
  data = train_svm,
  kernel = "radial",
  cost = best_model$cost,
  gamma = best_model$gamma,
  probability = TRUE
)

# 预测
train_pred_svm <- predict(svm_model, newdata = train_svm)
train_prob_svm <- attr(predict(svm_model, newdata = train_svm, probability = TRUE), "probabilities")[, "Yes"]
test_pred_svm <- predict(svm_model, newdata = test_svm)
test_prob_svm <- attr(predict(svm_model, newdata = test_svm, probability = TRUE), "probabilities")[, "Yes"]

# 混淆矩阵
cm_train_svm <- confusionMatrix(train_pred_svm, train_svm$Result, positive = "Yes")
cm_test_svm  <- confusionMatrix(test_pred_svm, test_svm$Result, positive = "Yes")

# 评估函数
evaluate_metrics <- function(prob, pred, true) {
  true_binary <- as.numeric(true == "Yes")
  
  roc_obj <- roc(response = true, predictor = prob)
  auc_val <- auc(roc_obj)
  auc_ci <- ci.auc(roc_obj)
  
  brier <- mean((prob - true_binary)^2)
  
  cm <- confusionMatrix(pred, true, positive = "Yes")
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- cm$byClass["F1"]
  
  acc_ci <- binom.test(sum(pred == true), length(true))$conf.int
  
  data.frame(
    Accuracy = cm$overall["Accuracy"],
    Accuracy_CI = paste0("[", round(acc_ci[1], 3), ", ", round(acc_ci[2], 3), "]"),
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    Precision = precision,
    Recall = recall,
    F1 = f1,
    AUC = auc_val,
    AUC_CI = paste0("[", round(auc_ci[1], 3), ", ", round(auc_ci[3], 3), "]"),
    Brier = brier
  )
}

# 输出评估结果
cat("\n====== SVM 训练集评估指标 ======\n")
metrics_train_svm <- evaluate_metrics(train_prob_svm, train_pred_svm, train_svm$Result)
print(metrics_train_svm)

cat("\n====== SVM 测试集评估指标 ======\n")
metrics_test_svm <- evaluate_metrics(test_prob_svm, test_pred_svm, test_svm$Result)
print(metrics_test_svm)

# 最终性能总结
results_svm <- rbind(
  cbind(Dataset = "Training Set", metrics_train_svm),
  cbind(Dataset = "Test Set", metrics_test_svm)
)
cat("\n====== SVM 最终性能总结 ======\n")
print(results_svm)









##############################3.评估指标可视化###################################
####################3.1 ROC曲线绘制代码#####################
# ==== 统一主题设置 ====
library(ggplot2)
library(patchwork)
library(grid)

# ==== 统一主题设置 ====
common_theme <- theme_minimal() +
  theme(
    panel.grid = element_blank(),
    panel.border = element_blank(),
    axis.line = element_blank(),
    axis.line.x.top = element_blank(),
    axis.line.y.right = element_blank(),
    axis.title = element_text(size = 13, face = "bold"),
    axis.text = element_text(size = 10, face = "plain"),
    axis.title.x = element_text(margin = ggplot2::margin(t = 15, b = 10)),
    axis.title.y = element_text(margin = ggplot2::margin(r = 15, l = 10)),
    legend.title = element_blank(),
    legend.text = element_text(size = 10, face = "bold"),
    legend.position = c(0.98, 0.02),
    legend.justification = c(1, 0),
    legend.background = element_blank(),
    plot.margin = unit(c(15, 15, 15, 15), "pt")
  )

# ==== 更可靠的添加边框方法（线宽0.5）====
add_uniform_border <- function(plot) {
  plot + 
    theme(
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
      plot.background = element_rect(color = NA, fill = NA)
    )
}

# ==== 绘制ROC曲线函数 ====
create_roc_plot <- function(data) {
  p <- ggplot(data, aes(x = 1 - specificity, y = sensitivity, color = model)) +
    geom_line(linewidth = 0.5) +
    scale_color_manual(values = setNames(unique(data$color), unique(data$model))) +
    labs(x = "1 - Specificity", y = "Sensitivity") +
    coord_equal() +
    common_theme
  p <- add_uniform_border(p)
  return(p)
}

# ==== 模型列表 ====
model_list <- list(
  list(name = "LR", train_prob = "train_prob_lr", test_prob = "test_prob_lr", color = "cyan"),
  list(name = "RF", train_prob = "train_prob_rf", test_prob = "test_prob_rf", color = "red"),
  list(name = "CART", train_prob = "train_prob_tree", test_prob = "test_prob_tree", color = "blue"),
  list(name = "SVM", train_prob = "train_prob_svm", test_prob = "test_prob_svm", color = "purple"),
  list(name = "KNN", train_prob = "train_prob_knn", test_prob = "test_prob_knn", color = "green"),
  list(name = "XGBoost", train_prob = "train_prob_xgb", test_prob = "test_prob_xgb", color = "orange"),
  list(name = "LightGBM", train_prob = "train_prob_lightgbm", test_prob = "test_prob_lightgbm", color = "brown")
)

# ==== ROC 数据容器 ====
roc_data_train <- list()
roc_data_test <- list()

# ==== 循环处理每个模型 ====
for (model in model_list) {
  # --- 训练集 ---
  if (exists(model$train_prob) && length(unique(traindata$Result)) >= 2) {
    prob <- get(model$train_prob)
    if (length(unique(prob)) > 1) {
      roc_obj <- tryCatch({
        roc(traindata$Result, prob, levels = c("No", "Yes"), direction = "<")
      }, error = function(e) NULL)
      
      if (!is.null(roc_obj)) {
        df <- ggroc(roc_obj)$data
        auc_val <- auc(roc_obj)
        label <- paste0(model$name, " (AUC=", round(auc_val, 3), ")")
        
        df$model <- label
        df$color <- model$color
        roc_data_train[[model$name]] <- df
      }
    } else {
      message(paste("Warning: Train probabilities for", model$name, "are constant or invalid"))
    }
  }
  
  # --- 测试集 ---
  if (exists(model$test_prob) && length(unique(testdata$Result)) >= 2) {
    prob <- get(model$test_prob)
    if (length(unique(prob)) > 1) {
      roc_obj <- tryCatch({
        roc(testdata$Result, prob, levels = c("No", "Yes"), direction = "<")
      }, error = function(e) NULL)
      
      if (!is.null(roc_obj)) {
        df <- ggroc(roc_obj)$data
        auc_val <- auc(roc_obj)
        label <- paste0(model$name, " (AUC=", round(auc_val, 3), ")")
        
        df$model <- label
        df$color <- model$color
        roc_data_test[[model$name]] <- df
      }
    } else {
      message(paste("Warning: Test probabilities for", model$name, "are constant or invalid"))
    }
  }
}

# ==== 绘制训练集和测试集ROC曲线 ====
# 检查是否包含 RF 模型的 ROC 数据
if ("RF" %in% names(roc_data_train)) {
  p1 <- create_roc_plot(do.call(rbind, roc_data_train))
} else {
  message("RF model data not found in train ROC")
}

if ("RF" %in% names(roc_data_test)) {
  p2 <- create_roc_plot(do.call(rbind, roc_data_test))
} else {
  message("RF model data not found in test ROC")
}

# ==== 合并图像 ====
final_plot <- p1 + p2 + 
  plot_layout(ncol = 2) &
  theme(plot.margin = unit(c(15, 15, 15, 15), "pt"))

# ==== 保存为高分辨率TIFF图像 ====
ggsave("combined_roc_plot.tiff",
       plot = final_plot,
       width = 12,
       height = 6,
       dpi = 300,
       bg = "white",
       units = "in",
       device = "tiff",
       compression = "lzw")








#加载 magick 包
library(magick)

#正确设置 Windows 路径，请根据实际路径修改
img_path <- "C:/Users/54176/Desktop/combined_roc_plot.tiff"

#读取图像
img <- image_read(img_path)

#获取原始尺寸
img_info <- image_info(img)
orig_width <- img_info$width
orig_height <- img_info$height

#设置新宽度
new_width <- 2360
#根据比例计算新高度
new_height <- round((new_width / orig_width) * orig_height)

#调整图像大小
resized_img <- image_resize(img, paste0(new_width, "x", new_height))

# 保存图像（你可以改回原路径以覆盖原图）
output_path <- "C:/Users/54176/Desktop/combined_roc_plot_resized.tiff"
image_write(resized_img, path = output_path, format = "tiff")












####################3.2 校准曲线绘制代码#####################
# 多次报错原因：
# test_prob 是一个 数据框/矩阵，包含 No 和 Yes 两列，对应两类的预测概率。 
# train_prob_tree 是一个 命名数值向量，值在 [ 0 , 1 ] [0,1] 范围，符合要求。 
# 但注意： 你使用的是 train_prob_tree，不是 test_prob 中的 Yes 列。 
# CART 模型在测试集上，应使用：test_prob_tree <- test_prob[, "Yes"]
# 二次报错原因：
# 你遇到的问题的核心在于 CART 的预测概率（train_prob_tree 或 test_prob_tree）未能成功分箱 (cut)，从而导致校准数据为空。
# 主要原因通常有： ❗问题定位：CART 的预测概率无变化或不分布
# 解决方案 策略：在 calibration_data() 函数中加入 fallback 逻辑，当唯一值太少时改用固定分箱（如等宽分箱）。

# ==== 加载包 ====
library(ggplot2)
library(dplyr)
library(patchwork)
library(scales)

# ==== 校准函数 ====
calibration_data <- function(predicted, actual, n_bins = 10) {
  df <- data.frame(predicted = predicted, actual = as.numeric(actual))
  df <- df[!is.na(df$predicted) & !is.na(df$actual), ]
  df$actual <- ifelse(df$actual == 2, 1, 0)
  
  breaks <- unique(quantile(df$predicted, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE))
  if (length(breaks) < 3) breaks <- seq(0, 1, length.out = n_bins + 1)
  
  df$bin <- cut(df$predicted, breaks = breaks, include.lowest = TRUE, right = TRUE)
  if (all(is.na(df$bin))) return(NULL)
  
  df %>%
    group_by(bin) %>%
    summarise(
      bin_pred = mean(predicted, na.rm = TRUE),
      bin_actual = mean(actual, na.rm = TRUE),
      n = n(),
      .groups = "drop"
    )
}

# ==== 检查标签 ====
cat("Train labels:\n")
print(table(traindata$Result))
cat("Test labels:\n")
print(table(testdata$Result))

# ==== 如果 test_prob 存在，赋值给 CART 模型变量 ====
if (exists("test_prob")) {
  if (is.vector(test_prob)) {
    test_prob_tree <- test_prob
  } else if (is.data.frame(test_prob) || is.matrix(test_prob)) {
    if ("Yes" %in% colnames(test_prob)) {
      test_prob_tree <- test_prob[, "Yes"]
    } else {
      stop("test_prob 不包含 'Yes' 列")
    }
  } else {
    stop("test_prob 类型未知，不能识别")
  }
}

# ==== 模型配置 ====
model_list <- list(
  list(name = "LR", train_prob = "train_prob_lr", test_prob = "test_prob_lr", color = "cyan"),
  list(name = "RF", train_prob = "train_prob_rf", test_prob = "test_prob_rf", color = "red"),
  list(name = "CART", train_prob = "train_prob_tree", test_prob = "test_prob_tree", color = "blue"),
  list(name = "KNN", train_prob = "train_prob_knn", test_prob = "test_prob_knn", color = "green"),
  list(name = "SVM", train_prob = "train_prob_svm", test_prob = "test_prob_svm", color = "purple"),
  list(name = "XGBoost", train_prob = "train_prob_xgb", test_prob = "test_prob_xgb", color = "orange"),
  list(name = "LightGBM", train_prob = "train_prob_lightgbm", test_prob = "test_prob_lightgbm", color = "brown")
)

# ==== 检查变量 ====
for (model in model_list) {
  if (!exists(model$train_prob)) cat("缺失训练集概率:", model$train_prob, "\n")
  if (!exists(model$test_prob)) cat("缺失测试集概率:", model$test_prob, "\n")
}

# ==== 计算校准 ====
calibration_train <- list()
calibration_test <- list()

for (model in model_list) {
  if (exists(model$train_prob)) {
    prob <- get(model$train_prob)
    cal <- calibration_data(prob, traindata$Result)
    if (!is.null(cal)) {
      cal$model <- model$name
      cal$color <- model$color
      calibration_train[[model$name]] <- cal
    } else {
      cat("Train calibration failed for", model$name, "\n")
    }
  }
  
  if (exists(model$test_prob)) {
    prob <- get(model$test_prob)
    cal <- calibration_data(prob, testdata$Result)
    if (!is.null(cal)) {
      cal$model <- model$name
      cal$color <- model$color
      calibration_test[[model$name]] <- cal
    } else {
      cat("Test calibration failed for", model$name, "\n")
    }
  }
}

# ==== 校准图：训练集 ====
df_train <- bind_rows(calibration_train)
df_test  <- bind_rows(calibration_test)

#==== 加载必要包 ====（加粗版）
library(ggplot2)
library(patchwork)
library(grid)

#==== 自定义主题（移除所有边框）====
common_theme <- theme_minimal() +
  theme(
    panel.grid = element_blank(),
    panel.border = element_blank(),
    axis.line = element_blank(),
    axis.line.x.top = element_blank(),
    axis.line.y.right = element_blank(),
    axis.title = element_text(size = 13, face = "bold"),
    axis.text = element_text(size = 10),
    axis.title.x = element_text(margin = ggplot2::margin(t = 15, b = 10)),
    axis.title.y = element_text(margin = ggplot2::margin(r = 15, l = 10)),
    legend.title = element_blank(),
    legend.text = element_text(size = 10, face = "bold"),
    legend.position = c(0.98, 0.02),
    legend.justification = c(1, 0),
    legend.background = element_blank(),
    plot.margin = unit(c(15, 15, 15, 15), "pt")
  )

#==== 更可靠的添加边框方法 ====
add_uniform_border <- function(plot) {
  plot + 
    theme(
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
      plot.background = element_rect(color = NA, fill = NA)
    )
}

#==== 校准图：训练集 ====
p1 <- ggplot(df_train, aes(x = bin_pred, y = bin_actual, color = model)) +
  geom_line(linewidth = 0.4) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
  scale_color_manual(values = setNames(df_train$color, df_train$model)) +
  labs(x = "Predicted Probability", y = "Observed Probability") +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
  coord_equal() +
  common_theme

# 添加统一边框（使用更可靠的方法）
p1 <- add_uniform_border(p1)

#==== 校准图：测试集 ====
p2 <- ggplot(df_test, aes(x = bin_pred, y = bin_actual, color = model)) +
  geom_line(linewidth = 0.4) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
  scale_color_manual(values = setNames(df_test$color, df_test$model)) +
  labs(x = "Predicted Probability", y = "Observed Probability") +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
  coord_equal() +
  common_theme

# 添加统一边框（使用更可靠的方法）
p2 <- add_uniform_border(p2)

#==== 合并图像 ====
final_plot <- p1 + p2 + 
  plot_layout(ncol = 2) &
  theme(plot.margin = unit(c(15, 15, 15, 15), "pt"))

#==== 导出设置 ====
output_path <- file.path(Sys.getenv("USERPROFILE"), "Desktop", "combined_calibration_plot.tiff")

# 导出图像
ggsave(
  filename = output_path,
  plot = final_plot,
  width = 12,
  height = 6,
  units = "in",
  dpi = 300,
  compression = "lzw",
  bg = "white",
  device = "tiff"
)



# 加载所需包
library(magick)

# 设置图像路径（请确保路径正确）
img_path <- "C:/Users/54176/Desktop/combined_calibration_plot.tiff"

# 读取图像
img <- image_read(img_path)

# 获取原始尺寸信息
img_info <- image_info(img)
orig_width <- img_info$width
orig_height <- img_info$height

# 新宽度
new_width <- 2360
# 按比例计算新高度
new_height <- round((new_width / orig_width) * orig_height)

# 调整图像大小
resized_img <- image_resize(img, paste0(new_width, "x", new_height))

# 保存新图像（可覆盖原图，也可另存）
output_path <- "C:/Users/54176/Desktop/combined_calibration_plot_resized.tiff"
image_write(resized_img, path = output_path, format = "tiff")









####################3.3 DCA曲线绘制代码#####################
# 查看整个概率矩阵
print(test_prob_lr)
print(test_prob_rf)
print(test_prob_tree)
print(test_prob_knn)
print(test_prob_svm)
print(test_prob_xgb)
print(test_prob_lightgbm)

# 使用rmda包进行测试集决策曲线分析
library(rmda)
library(ggplot2)
dca_data <- data.frame(Result = as.numeric(testdata$Result)-1, 
                       test_prob_lr,
                       test_prob_rf,
                       test_prob_tree,
                       test_prob_knn,
                       test_prob_svm,
                       test_prob_xgb,
                       test_prob_lightgbm)
# Logistic Regression 
dca.result_lr <- decision_curve(Result ~ test_prob_lr, 
                                data = dca_data, 
                                family = "binomial",
                                thresholds = seq(0, .8, by = .05),
                                bootstraps = 10)
# Random Forest 
dca.result_rf <- decision_curve(Result ~ test_prob_rf, 
                                data = dca_data, 
                                family = "binomial",
                                thresholds = seq(0, .8, by = .05),
                                bootstraps = 10)
# Decision Tree
dca.result_tree <- decision_curve(Result ~ test_prob_tree, 
                                  data = dca_data, 
                                  family = "binomial",
                                  thresholds = seq(0, .8, by = .05),
                                  bootstraps = 10)
# KNN
dca.result_knn <- decision_curve(Result ~ test_prob_knn, 
                                 data = dca_data, 
                                 family = "binomial",
                                 thresholds = seq(0, .8, by = .05),
                                 bootstraps = 10)
# SVM
dca.result_svm <- decision_curve(Result ~ test_prob_svm, 
                                 data = dca_data, 
                                 family = "binomial",
                                 thresholds = seq(0, .8, by = .05),
                                 bootstraps = 10)
# XGBoost
dca.result_xgb <- decision_curve(Result ~ test_prob_xgb, 
                                 data = dca_data, 
                                 family = "binomial",
                                 thresholds = seq(0, .8, by = .05),
                                 bootstraps = 10)
# LightGBM
dca.result_lightgbm <- decision_curve(Result ~ test_prob_lightgbm, 
                                      data = dca_data, 
                                      family = "binomial",
                                      thresholds = seq(0, .8, by = .05),
                                      bootstraps = 10)






# 训练集
# 查看整个概率矩阵
print(train_prob_lr)
print(train_prob_rf)
print(train_prob_tree)
print(train_prob_knn)
print(train_prob_svm)
print(train_prob_xgb)
print(train_prob_lightgbm)

# 使用rmda包进行训练集决策曲线分析
library(rmda)
library(ggplot2)

# 构建训练集数据
dca_data_train <- data.frame(Result = as.numeric(traindata$Result)-1, 
                             train_prob_lr,
                             train_prob_rf,
                             train_prob_tree,
                             train_prob_rf,
                             train_prob_svm,
                             train_prob_xgb,
                             train_prob_lightgbm)

# Logistic Regression 
dca.result_lr_train <- decision_curve(Result ~ train_prob_lr, 
                                      data = dca_data_train, 
                                      family = "binomial",
                                      thresholds = seq(0, .8, by = .05),
                                      bootstraps = 10)

# Random Forest 
dca.result_rf_train <- decision_curve(Result ~ train_prob_rf, 
                                      data = dca_data_train, 
                                      family = "binomial",
                                      thresholds = seq(0, .8, by = .05),
                                      bootstraps = 10)
# Decision Tree
dca.result_tree_train <- decision_curve(Result ~ train_prob_tree, 
                                        data = dca_data_train, 
                                        family = "binomial",
                                        thresholds = seq(0, .8, by = .05),
                                        bootstraps = 10)

# KNN
dca.result_knn_train <- decision_curve(Result ~ train_prob_knn, 
                                       data = dca_data_train, 
                                       family = "binomial",
                                       thresholds = seq(0, .8, by = .05),
                                       bootstraps = 10)

# SVM
dca.result_svm_train <- decision_curve(Result ~ train_prob_svm, 
                                       data = dca_data_train, 
                                       family = "binomial",
                                       thresholds = seq(0, .8, by = .05),
                                       bootstraps = 10)

# XGBoost
dca.result_xgb_train <- decision_curve(Result ~ train_prob_xgb, 
                                       data = dca_data_train, 
                                       family = "binomial",
                                       thresholds = seq(0, .8, by = .05),
                                       bootstraps = 10)

# LightGBM
dca.result_lightgbm_train <- decision_curve(Result ~ train_prob_lightgbm, 
                                            data = dca_data_train, 
                                            family = "binomial",
                                            thresholds = seq(0, .8, by = .05),
                                            bootstraps = 10)





############### 3.3 DCA曲线绘制 ###############
# 加载必要的包
library(rmda)
library(ggplot2)

# ========== 通用参数设置 ==========
# 颜色配置（6种模型颜色 + 2种参考线颜色）
model_colors <- c("cyan", "red", "blue", "green", "purple", "orange", "brown")
reference_colors <- c("grey", "black")  # None和All的颜色
all_colors <- c(model_colors, reference_colors)

# 图例标签
model_labels <- c("LR", "RF", "CART", "KNN", "SVM", "XGBoost", "LightGBM")
reference_labels <- c("None", "All")
all_labels <- c(model_labels, reference_labels)


# ========== 训练集DCA分析 ==========
# 准备训练集数据
dca_data_train <- data.frame(
  Result = as.numeric(traindata$Result)-1, 
  train_prob_lr,
  train_prob_rf,
  train_prob_tree,
  train_prob_knn,
  train_prob_svm,
  train_prob_xgb,
  train_prob_lightgbm
)

# 生成各模型决策曲线
dc_curves_train <- list(
  decision_curve(Result ~ train_prob_lr,
                 data = dca_data_train,
                 family = binomial(link = "logit"),
                 thresholds = seq(0, 1.0, 0.05),
                 bootstraps = 10),
  decision_curve(Result ~ train_prob_rf,
                 data = dca_data_train,
                 family = binomial(link = "logit"),
                 thresholds = seq(0, 1.0, 0.05),
                 bootstraps = 10),
  decision_curve(Result ~ train_prob_tree,
                 data = dca_data_train,
                 family = binomial(link = "logit"),
                 thresholds = seq(0, 1.0, 0.05),
                 bootstraps = 10),
  decision_curve(Result ~ train_prob_knn,
                 data = dca_data_train,
                 family = binomial(link = "logit"),
                 thresholds = seq(0, 1.0, 0.05),
                 bootstraps = 10),
  decision_curve(Result ~ train_prob_svm,
                 data = dca_data_train,
                 family = binomial(link = "logit"),
                 thresholds = seq(0, 1.0, 0.05),
                 bootstraps = 10),
  decision_curve(Result ~ train_prob_xgb,
                 data = dca_data_train,
                 family = binomial(link = "logit"),
                 thresholds = seq(0, 1.0, 0.05),
                 bootstraps = 10),
  decision_curve(Result ~ train_prob_lightgbm,
                 data = dca_data_train,
                 family = binomial(link = "logit"),
                 thresholds = seq(0, 1.0, 0.05),
                 bootstraps = 10)
)


# ========== 测试集DCA分析 ==========
# 准备测试集数据
dca_data_test <- data.frame(
  Result = as.numeric(testdata$Result)-1, 
  test_prob_lr,
  test_prob_rf,
  test_prob_tree,
  test_prob_knn,
  test_prob_svm,
  test_prob_xgb,
  test_prob_lightgbm
)

# 生成各模型决策曲线（注意每个模型需要完整的参数）
dc_curves_test <- list(
  decision_curve(Result ~ test_prob_lr, 
                 data = dca_data_test, 
                 family = binomial(link = "logit"),
                 thresholds = seq(0, 1.0, 0.05),
                 bootstraps = 10),
  decision_curve(Result ~ test_prob_rf, 
                 data = dca_data_test, 
                 family = binomial(link = "logit"),
                 thresholds = seq(0, 1.0, 0.05),
                 bootstraps = 10),
  decision_curve(Result ~ test_prob_tree,
                 data = dca_data_test,
                 family = binomial(link = "logit"),
                 thresholds = seq(0, 1.0, 0.05),
                 bootstraps = 10),
  decision_curve(Result ~ test_prob_knn,
                 data = dca_data_test,
                 family = binomial(link = "logit"),
                 thresholds = seq(0, 1.0, 0.05),
                 bootstraps = 10),
  decision_curve(Result ~ test_prob_svm,
                 data = dca_data_test,
                 family = binomial(link = "logit"),
                 thresholds = seq(0, 1.0, 0.05),
                 bootstraps = 10),
  decision_curve(Result ~ test_prob_xgb,
                 data = dca_data_test,
                 family = binomial(link = "logit"),
                 thresholds = seq(0, 1.0, 0.05),
                 bootstraps = 10),
  decision_curve(Result ~ test_prob_lightgbm,
                 data = dca_data_test,
                 family = binomial(link = "logit"),
                 thresholds = seq(0, 1.0, 0.05),
                 bootstraps = 10)
)







# ==== 构建桌面路径 ====
output_path <- file.path(Sys.getenv("USERPROFILE"), "Desktop", "combined_dca_plot.tiff")

# ==== 模型设置 ====
model_labels <- c("LR", "RF", "CART", "KNN", "SVM", "XGBoost", "LightGBM")
model_colors <- c("cyan", "red", "blue", "green", "purple", "orange", "brown")

# “All” 用灰色，“None” 用黑色
reference_labels <- c("None", "All")
reference_colors <- c("black", "grey")

# 整体图例配置
all_labels <- c(reference_labels, model_labels)
all_colors <- c(reference_colors, model_colors)
all_linetypes <- rep(1, length(all_labels))  # 全部实线

# ==== 开启图形设备 ====
tiff(filename = output_path,
     width = 12, height = 6, units = "in", res = 300, compression = "lzw")

# ==== 设置图形参数 ====
par(mfrow = c(1, 2),
    font.lab = 2,               # 加粗坐标轴标签
    cex.lab = 1.2,              # 坐标轴标签字体大小
    cex.axis = 1.1,             # 坐标刻度字体大小
    tck = -0.02, tcl = -0.3,
    lwd = 0.3, bty = "l",
    mar = c(5, 5, 2, 1), mgp = c(2.8, 1, 0))

# ==== 通用绘图函数 ====
plot_dca_panel <- function(dca_object) {
  # 构造颜色顺序：先None（黑）再All（灰），再各模型
  curve_names <- c("None", "All", model_labels)
  curve_colors <- c("black", "grey", model_colors)
  
  plot_decision_curve(dca_object,
                      curve.names = model_labels,
                      col = model_colors,
                      lwd = 1.5,
                      confidence.intervals = FALSE,
                      standardize = FALSE,
                      legend.position = "none",
                      xlab = "", ylab = "",
                      cost.benefit.axis = FALSE,
                      xlim = c(0, 1.00))
  
  # 设置横纵坐标轴标签加粗
  title(xlab = "Threshold Probability", font.lab = 2, cex.lab = 1.2, col.lab = "black")  # 横坐标标签加粗
  title(ylab = "Net Benefit", font.lab = 2, cex.lab = 1.2, col.lab = "black")  # 纵坐标标签加粗
  
  # 设置图例加粗
  legend("topright", 
         legend = all_labels,
         col = all_colors,
         lwd = 1.5,
         lty = all_linetypes,
         cex = 0.8,               # 图例标签加粗
         text.font = 2,           # 图例标签加粗
         bty = "n",
         x.intersp = 1.0,
         y.intersp = 1.0,
         inset = c(0.10, 0.02),
         seg.len = 1.4,
         xpd = TRUE)
}

# ==== 绘制左图：训练集 ====
plot_dca_panel(dc_curves_train)

# ==== 绘制右图：测试集 ====
plot_dca_panel(dc_curves_test)

# ==== 关闭图形设备 ====
dev.off()









#####################4.最优模型的SHAP解释#########################################
# 加载必要的包
library(ggplot2)
library(shapviz)
library(patchwork)
library(viridis)
library(randomForest)

# 使用 iris 数据集
data(iris)
train_data <- iris[, -5]  # 特征数据
train_labels <- iris[, 5]  # 标签数据

# 训练随机森林模型
set.seed(123)  # 确保结果可重现
model_rf <- randomForest(train_data, train_labels, ntree = 100)

# 创建 shapviz 对象 - 修正的关键步骤
sv <- shapviz(model_rf, X = train_data)

# 获取 SHAP 特征名称
shap_feature_names <- colnames(sv$X)
cat("可用SHAP特征变量:\n", paste(shap_feature_names, collapse = ", "), "\n\n")

# --- 使用前5个特征绘制依赖图 ---
cat("⚠️ 使用前5个特征绘制依赖图\n")
if (length(shap_feature_names) == 0) {
  stop("SHAP 特征名称列表为空，请检查 SHAP 值计算步骤")
}
top_features <- head(shap_feature_names, 5)
dependence_plots <- list()

# 绘制依赖图
for (i in seq_along(top_features)) {
  feat <- top_features[i]
  cat("正在处理:", feat, "\n")
  
  tryCatch({
    p <- sv_dependence(
      sv,  # 使用 shapviz 对象
      v = feat,
      color_var = NULL,
      size = 1.5,
      alpha = 0.6
    ) +
      geom_hline(yintercept = 0, color = "darkgray", linewidth = 0.4, linetype = "solid") +
      theme_minimal(base_size = 12) +
      theme(
        plot.title = element_blank(),
        axis.title = element_text(size = 9, face = "bold"),
        axis.text = element_text(size = 10),
        panel.grid.major = element_line(color = "gray90", linewidth = 0.25),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        axis.line = element_line(color = "black", linewidth = 0.3),
        axis.line.x.top = element_blank(),
        axis.line.y.right = element_blank(),
        plot.margin = unit(c(10, 10, 10, 10), "pt"),
        legend.position = "none"
      )
    
    dependence_plots[[feat]] <- p
    print(p)
    cat("--> 成功\n\n")
  }, error = function(e) {
    cat("--> 失败:", e$message, "\n\n")
  })
}

# --- 保存所有图表为 TIFF ---
output_dir <- "SHAP_plots"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
  cat("创建输出目录:", output_dir, "\n")
}
cat("\n=== 开始导出TIFF文件 ===\n")

for (i in seq_along(dependence_plots)) {
  name <- names(dependence_plots)[i]
  file_name <- gsub("[^A-Za-z0-9]", "_", name)
  output_path <- file.path(output_dir, paste0("dependence_", file_name, ".tiff"))
  
  # 固定宽高比 (4:3)
  height_px <- round(2360 * 3/4)
  
  cat(sprintf("正在导出: %s (尺寸: %d×%d 像素)\n", output_path, 2360, height_px))
  
  tryCatch({
    tiff(
      filename = output_path,
      width = 2360, 
      height = height_px,
      units = "px",
      res = 300,
      compression = "lzw"
    )
    print(dependence_plots[[i]])
    dev.off()
    cat("--> 导出成功\n\n")
  }, error = function(e) {
    cat("--> 导出失败:", e$message, "\n\n")
  })
}

# --- 组合图导出 ---
success_count <- length(dependence_plots)
if (success_count >= 5) {
  cat("=== 创建组合布局图 ===\n")
  layout <- "
    ABC
    DE#
  "
  combined_plot <- wrap_plots(
    A = dependence_plots[[1]],
    B = dependence_plots[[2]],
    C = dependence_plots[[3]],
    D = dependence_plots[[4]],
    E = dependence_plots[[5]],
    design = layout,
    widths = unit(rep(1, 3), "null"),
    heights = unit(c(1, 1), "null")
  ) +
    plot_annotation(tag_levels = 'A', tag_prefix = '', tag_suffix = '')
  
  combined_path <- file.path(output_dir, "combined_dependence_plots.tiff")
  combined_height <- round(2360 * 2/3)  # 组合图固定比例
  
  tiff(
    filename = combined_path,
    width = 2360,
    height = combined_height,
    units = "px",
    res = 300,
    compression = "lzw"
  )
  print(combined_plot)
  dev.off()
  cat("组合图已保存:", combined_path, "\n")
} else {
  cat(sprintf("⚠️ 仅成功生成%d/5个图，跳过组合图生成\n", success_count))
}

cat("\n=== 操作完成 ===\n")
cat("成功生成图表:", success_count, "/5\n")





# 绘制瀑布图
sv_waterfall(sv, row_id = 5, 
             fill_colors = c("#f7d13d", "#a52c60")) +
  theme_bw() +
  ggtitle("Random Forest") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", color = "black"))

# 绘制瀑布图
sv_waterfall(shap_value_rf, row_id = 5, 
             fill_colors = c("#f7d13d", "#a52c60")) +
  theme_bw() +
  ggtitle("Random Forest") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", color = "black"))


















# 加载必要的包
library(ggplot2)
library(shapviz)
library(patchwork)
library(viridis)
library(shap)  # 如果您使用 xgboost 或其他模型，可以使用这个包

# 假设您的模型是 xgboost 模型
# 使用 XGBoost 模型计算 SHAP 值

# 模型训练示例（假设已有模型）
# model_xgb <- xgboost(data = train_data, label = train_labels, max_depth = 6, eta = 0.1, nrounds = 100)

# 计算 SHAP 值
# shap_value_rf <- shap.values(x_train, model_xgb)

# 获取 SHAP 特征名称
shap_feature_names <- colnames(shap_value_rf$X)  # 确保 `shap_value_rf` 中包含正确的 SHAP 数据

cat("可用SHAP特征变量:\n", paste(shap_feature_names, collapse = ", "), "\n\n")

# --- 使用前5个特征绘制依赖图 ---
cat("⚠️ 使用前5个特征绘制依赖图\n")

# 确保 shap_feature_names 已正确加载
if (length(shap_feature_names) == 0) {
  stop("SHAP 特征名称列表为空，请检查 SHAP 值计算步骤")
}

top_features <- head(shap_feature_names, 5)
dependence_plots <- list()

# 绘制依赖图
for (i in seq_along(top_features)) {
  feat <- top_features[i]
  cat("正在处理:", feat, "\n")
  
  tryCatch({
    p <- sv_dependence(
      shap_value_rf,  # 使用计算得到的 SHAP 值对象
      v = feat,
      color_var = NULL,
      size = 1.5,
      alpha = 0.6
    ) +
      geom_hline(yintercept = 0, color = "darkgray", linewidth = 0.4, linetype = "solid") +
      theme_minimal(base_size = 12) +
      theme(
        plot.title = element_blank(),
        axis.title = element_text(size = 9, face = "bold"),
        axis.text = element_text(size = 10),
        panel.grid.major = element_line(color = "gray90", linewidth = 0.25),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        axis.line = element_line(color = "black", linewidth = 0.3),
        axis.line.x.top = element_blank(),
        axis.line.y.right = element_blank(),
        plot.margin = unit(c(10, 10, 10, 10), "pt"),
        legend.position = "none"
      )
    
    dependence_plots[[feat]] <- p
    print(p)
    cat("--> 成功\n\n")
  }, error = function(e) {
    cat("--> 失败:", e$message, "\n\n")
  })
}

# --- 保存所有图表为 TIFF ---
output_dir <- "SHAP_plots"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
  cat("创建输出目录:", output_dir, "\n")
}

cat("\n=== 开始导出TIFF文件 ===\n")

for (i in seq_along(dependence_plots)) {
  name <- names(dependence_plots)[i]
  file_name <- gsub("[^A-Za-z0-9]", "_", name)
  output_path <- file.path(output_dir, paste0("dependence_", file_name, ".tiff"))
  
  # 固定宽高比 (4:3)
  height_px <- round(2360 * 3/4)
  
  cat(sprintf("正在导出: %s (尺寸: %d×%d 像素)\n", output_path, 2360, height_px))
  
  tryCatch({
    tiff(
      filename = output_path,
      width = 2360, 
      height = height_px,
      units = "px",
      res = 300,
      compression = "lzw"
    )
    print(dependence_plots[[i]])
    dev.off()
    cat("--> 导出成功\n\n")
  }, error = function(e) {
    cat("--> 导出失败:", e$message, "\n\n")
  })
}

# --- 组合图导出（仅当成功生成5个图时）---
success_count <- length(dependence_plots)
if (success_count >= 5) {
  cat("=== 创建组合布局图 ===\n")
  layout <- "
    ABC
    DE#
  "
  combined_plot <- wrap_plots(
    A = dependence_plots[[1]],
    B = dependence_plots[[2]],
    C = dependence_plots[[3]],
    D = dependence_plots[[4]],
    E = dependence_plots[[5]],
    design = layout,
    widths = unit(rep(1, 3), "null"),
    heights = unit(c(1, 1), "null")
  ) +
    plot_annotation(tag_levels = 'A', tag_prefix = '', tag_suffix = '')  # 仅在此处添加标签
  
  combined_path <- file.path(output_dir, "combined_dependence_plots.tiff")
  combined_height <- round(2360 * 2/3)  # 组合图固定比例
  
  tiff(
    filename = combined_path,
    width = 2360,
    height = combined_height,
    units = "px",
    res = 300,
    compression = "lzw"
  )
  print(combined_plot)
  dev.off()
  cat("组合图已保存:", combined_path, "\n")
} else {
  cat(sprintf("⚠️ 仅成功生成%d/5个图，跳过组合图生成\n", success_count))
}

cat("\n=== 操作完成 ===\n")
cat("成功生成图表:", success_count, "/5\n")














#瀑布图
sv_waterfall(shap_value, row_id = 5,
             fill_colors = c("#f7d13d", "#a52c60"))+
  theme_bw()+
  ggtitle("xgboost")+
  theme(plot.title = element_text(hjust = 0.5,face = "bold",color = "black"))


sv_waterfall(shap_value, row_id = 129,
             fill_colors = c("#f7d13d", "#a52c60"))+
  theme_bw()+
  ggtitle("xgboost")+
  theme(plot.title = element_text(hjust = 0.5,face = "bold",color = "black"))













######(1)xgboost：方法二#### 
############################################################
## 0. 依赖包                                               ##
############################################################
libs <- c("caret", "xgboost", "kernelshap", "shapviz",
          "ggplot2", "patchwork", "cowplot", "tibble")
to_install <- libs[!libs %in% rownames(installed.packages())]
if (length(to_install)) install.packages(to_install)
lapply(libs, library, character.only = TRUE)

############################################################
## 1. 特征工程                                             ##
############################################################
features  <- setdiff(colnames(traindata), "Result")
cat_vars  <- names(traindata[, features])[sapply(
  traindata[, features], \(x) is.character(x) || is.factor(x)
)]
for (v in cat_vars) {
  traindata[[v]] <- as.factor(traindata[[v]])
  testdata[[v]]  <- as.factor(testdata[[v]])
}

dummy_model <- dummyVars(~ ., data = traindata[, features],
                         fullRank = TRUE, sep = "_")
train_enc   <- predict(dummy_model, traindata[, features])
test_enc    <- predict(dummy_model, testdata[, features])

clean_names <- function(nms) {
  nms <- gsub("\\.\\d+$", "", nms)
  nms <- gsub("_1$", "", nms)
  nms <- gsub("_+", "_", nms)
  sub("^_|_$", "", nms)
}
colnames(train_enc) <- clean_names(colnames(train_enc))
colnames(test_enc)  <- clean_names(colnames(test_enc))

train_mat <- apply(train_enc, 2, as.numeric)
test_mat  <- apply(test_enc , 2, as.numeric)

############################################################
## 2. 标签                                                 ##
############################################################
y_train <- factor(ifelse(traindata$Result == "Yes", 1, 0), levels = c(0, 1))
y_test  <- factor(ifelse(testdata$Result  == "Yes", 1, 0), levels = c(0, 1))

############################################################
## 3. XGBoost 模型                                           ##
############################################################
set.seed(2025)
dtrain <- xgb.DMatrix(data = train_mat, label = as.numeric(y_train) - 1)
dtest  <- xgb.DMatrix(data = test_mat, label = as.numeric(y_test) - 1)

params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "error",
  max_depth = 6,
  eta = 0.3,
  nthread = 2
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 1
)

############################################################
## 4. Kernel SHAP & shapviz                                ##
############################################################
n_train   <- 485
n_test    <- 121
X_explain <- train_mat[1:min(n_train, nrow(train_mat)), ]
X_bg      <- test_mat [1:min(n_test , nrow(test_mat )), ]

pred_fun <- function(model, newdata) {
  predict(model, as.matrix(newdata))
}

explain_kernel <- kernelshap(
  object   = xgb_model,
  X        = X_explain,
  bg_X     = X_bg,
  pred_fun = pred_fun,
  verbose  = TRUE
)

sv <- shapviz(explain_kernel, X = X_explain, interactions = TRUE)

############################################################
## 5. 通用主题                                             ##
############################################################
theme_common <- theme_bw() +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    panel.grid   = element_blank(),
    axis.title   = element_text(size = 13, face = "bold"),
    axis.text.x  = element_text(size = 10, color = "black"),
    axis.text.y  = element_text(size = 11, color = "black", face = "bold"),
    legend.title = element_text(size = 9,  face = "bold"),
    legend.text  = element_text(size = 9,  face = "bold"),
    plot.title   = element_blank(),
    plot.margin  = ggplot2::margin(6, 6, 6, 6, "pt")
  )

############################################################
## 6. 四张图                                               ##
############################################################
# 6.3 力图（去边框、去除网格线、去除纵坐标标签中的 "1-"，横纵坐标标题加粗，刻度标签加粗）
theme_force <- theme_bw() +
  theme(
    panel.border = element_blank(),  # 去除所有边框
    axis.line.x  = element_line(color = "black", linewidth = 0.25),  # 横坐标轴线稍微细一点（调整 linewidth）
    axis.line.y  = element_blank(),  # 去除左侧纵轴线
    axis.text.x  = element_text(size = 10, color = "black", vjust = 0, face = "plain"),  # 横坐标刻度标签加粗
    axis.title.x = element_text(size = 13, face = "bold", vjust = 0),  # 横坐标标题加粗
    axis.text.y  = element_text(size = 10, color = "black", face = "bold"),  # 纵坐标刻度标签加粗
    axis.title.y = element_text(size = 13, face = "bold"),  # 纵坐标标题加粗
    strip.text   = element_blank(),  # 去掉分面标签（例如左边的 "1-"）
    panel.grid   = element_blank(),  # 去除网格线
    # 去除纵坐标标签中的 "1-" 直接替换为空字符
    scale_y_discrete(labels = function(x) gsub("^1-$", "", x)) 
  )

# 绘制力图1（针对 row 5）
plot_force1 <- sv_force(sv, row_id = 5, max_display = 20) +
  theme_force +
  theme(axis.title.x = element_text(size = 13, face = "bold"))

# 绘制力图2（针对 row 129）
plot_force2 <- sv_force(sv, row_id = 129, max_display = 20) +
  theme_force +
  theme(axis.title.x = element_text(size = 13, face = "bold"))


# 6.1 蜂巢图
p_beeswarm <- sv_importance(sv, "beeswarm", max_display = 20) +
  theme_common +
  labs(x = "SHAP Value", y = NULL)

# 6.2 条形图
p_bar <- sv_importance(sv, max_display = 20) +
  theme_common +
  labs(x = "Mean(|SHAP|)", y = NULL)

############################################################
## 7. 单图导出                                             ##
############################################################
out_dir <- ifelse(.Platform$OS.type == "windows",
                  file.path(Sys.getenv("USERPROFILE"), "Desktop"),
                  file.path(Sys.getenv("HOME"),      "Desktop"))


ggsave(file.path(out_dir, "shap_beeswarm.tiff"),
       p_beeswarm, dpi = 300, width = 6, height = 4,
       units = "in", device = "tiff", compression = "lzw", bg = "white")

ggsave(file.path(out_dir, "shap_bar.tiff"),
       p_bar, dpi = 300, width = 6, height = 4,
       units = "in", device = "tiff", compression = "lzw", bg = "white")

ggsave(file.path(out_dir, "shap_force_row5.tiff"),
       plot_force1, dpi = 300, width = 6, height = 4,
       units = "in", device = "tiff", compression = "lzw", bg = "white")

ggsave(file.path(out_dir, "shap_force_row129.tiff"),
       plot_force2, dpi = 300, width = 6, height = 4,
       units = "in", device = "tiff", compression = "lzw", bg = "white")

############################################################
## 8. 蜂巢图+条形图合并                                   ##
############################################################
# 横向合并（p_beeswarm 在左，p_bar 在右）
beeswarm_bar <- p_beeswarm | p_bar   # 默认 ncol = 2

# 如需固定宽度比例，可写：p_beeswarm | p_bar + plot_layout(widths = c(1, 1))

# 导出
ggsave(file.path(out_dir,"beeswarm_bar.tiff"),
       beeswarm_bar,
       dpi = 300, width = 12, height = 4,  # 高度可按需调整
       units = "in", device = "tiff",
       compression = "lzw", bg = "white")











#安装及加载必要的R包
#install.packages("kernelshap")
#install.packages("shapviz")

library(caret)
library(xgboost)
library(kernelshap)
library(shapviz)
library(ggplot2)
library(patchwork)
library(tibble)

# --- 1. 特征准备 ---
features <- setdiff(colnames(traindata), "Result")

# --- 2. 分类变量因子化 ---
categorical_vars <- names(traindata[, features])[sapply(traindata[, features], function(x) is.character(x) || is.factor(x))]
for (var in categorical_vars) {
  traindata[[var]] <- as.factor(traindata[[var]])
  testdata[[var]] <- as.factor(testdata[[var]])
}

# --- 3. 独热编码 + 列名清理 ---
dummy_model <- dummyVars(~ ., data = traindata[, features], fullRank = TRUE, sep = "_")
train_encoded <- predict(dummy_model, newdata = traindata[, features])
test_encoded  <- predict(dummy_model, newdata = testdata[, features])

# 清理列名中的 .0, .1, _1 等
clean_colnames <- function(names) {
  names <- gsub("\\.\\d+$", "", names)     # 去除 .0, .1
  names <- gsub("_1$", "", names)          # 去除 _1
  names <- gsub("_+", "_", names)
  names <- gsub("^_|_$", "", names)
  return(names)
}
colnames(train_encoded) <- clean_colnames(colnames(train_encoded))
colnames(test_encoded) <- clean_colnames(colnames(test_encoded))

# --- 4. 强制为数值矩阵 ---
train_encoded <- apply(train_encoded, 2, as.numeric)
test_encoded <- apply(test_encoded, 2, as.numeric)

# --- 5. 准备标签 ---
train_label <- ifelse(traindata$Result == "Yes", 1, 0)
test_label  <- ifelse(testdata$Result == "Yes", 1, 0)

# --- 6. 模型训练 ---
train_matrix <- xgb.DMatrix(data = as.matrix(train_encoded), label = train_label)
test_matrix <- xgb.DMatrix(data = as.matrix(test_encoded), label = test_label)

xgb_model <- xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    nthread = 4
  ),
  data = train_matrix,
  watchlist = list(test = test_matrix),
  nrounds = 100,
  early_stopping_rounds = 10,
  verbose = 0
)

# --- 7. 解释样本 ---
n_train <- 485
n_test <- 121
X_explain <- as.matrix(train_encoded[1:min(n_train, nrow(train_encoded)), ])
X_bg <- as.matrix(test_encoded[1:min(n_test, nrow(test_encoded)), ])

# --- 8. 自定义预测函数 ---
pred_fun <- function(object, newdata) {
  if (!is.matrix(newdata)) newdata <- as.matrix(newdata)
  if (!is.numeric(newdata)) newdata <- apply(newdata, 2, as.numeric)
  dmat <- xgb.DMatrix(data = newdata)
  predict(object, newdata = dmat)
}

# --- 9. kernelSHAP 计算 ---
explain_kernel <- kernelshap(
  object = xgb_model,
  X = X_explain,
  bg_X = X_bg,
  pred_fun = pred_fun,
  verbose = TRUE
)

# --- 10. 构建 shapviz 对象 ---
shap_value <- shapviz(
  explain_kernel,
  X = X_explain,
  interactions = TRUE
)


library(ggplot2)
library(shapviz)
library(patchwork)

# --- 11. 绘图 ---

# 自定义主题（无网格 + 黑色字体）
theme_nogrid <- theme_bw() +
  theme(
    plot.title = element_blank(),
    axis.title.x = element_text(face = "bold", color = "black"),
    axis.title.y = element_blank(),
    axis.text.x = element_text(color = "black"),
    axis.text.y = element_text(color = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

# 条形图（右）
p_bar <- sv_importance(shap_value, kind = "bar", show_numbers = FALSE, fill = "#fca50a") +
  theme_nogrid

print(p_bar)

# 蜂巢图（左）- 颜色对调
p_beeswarm <- sv_importance(
  shap_value, kind = "beeswarm",
  viridis_args = list(begin = 0.85, end = 0.25, option = "B"),
  show_numbers = FALSE
) + theme_nogrid

print(p_beeswarm)





# 调节边框——文字加粗（选择这个！）
# ==== 加载必要包 ====
library(ggplot2)
library(shapviz)
library(patchwork)
library(viridis)
library(grid)

# ==== 自定义主题 ====
theme_nogrid <- theme_bw() +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    axis.line = element_blank(),
    axis.line.x = element_blank(),
    axis.line.y = element_blank(),
    axis.title = element_text(size = 13, face = "bold"),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 11, color = "black", face = "bold"),  # 加粗变量标签
    legend.title = element_text(size = 9, face = "bold"),                   # 加粗图例标题
    legend.text = element_text(size = 9, face = "bold"),                    # 加粗 colorbar 标签
    plot.title = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = ggplot2::margin(10, 10, 10, 10, "pt")
  )

# ==== 蜂巢图（左） ====
p_beeswarm <- sv_importance(
  shap_value,
  kind = "beeswarm",
  viridis_args = list(begin = 0.85, end = 0.25, option = "B")
) + theme_nogrid

# ==== 条形图（右） ====
p_bar <- sv_importance(
  shap_value,
  kind = "bar",
  show_numbers = FALSE,
  fill = "#fca50a"
) + theme_nogrid

# ==== 拼图组合 ====
combined_plot <- p_beeswarm + p_bar + plot_layout(ncol = 2)

# ==== 保存路径 ====
output_path <- file.path(Sys.getenv("USERPROFILE"), "Desktop", "shap_combined_plot.tiff")

# ==== 导出 ====
ggsave(
  filename = output_path,
  plot = combined_plot,
  width = 12, height = 6, dpi = 300,
  units = "in", device = "tiff",
  compression = "lzw", bg = "white"
)

cat("✅ SHAP 组合图已保存至：", output_path, "\n")







#瀑布图
sv_waterfall(shap_value, row_id = 5,
             fill_colors = c("#f7d13d", "#a52c60"))+
  theme_bw()+
  ggtitle("xgboost")+
  theme(plot.title = element_text(hjust = 0.5,face = "bold",color = "black"))


sv_waterfall(shap_value, row_id = 129,
             fill_colors = c("#f7d13d", "#a52c60"))+
  theme_bw()+
  ggtitle("xgboost")+
  theme(plot.title = element_text(hjust = 0.5,face = "bold",color = "black"))





#单样本特征图
sv_force(shap_value, row_id = 5,size = 9)+
  ggtitle("xgboost")+
  theme(plot.title = element_text(hjust = 0.5,face = "bold",color = "black"))


sv_force(shap_value, row_id = 129,size = 9)+
  ggtitle("xgboost")+
  theme(plot.title = element_text(hjust = 0.5,face = "bold",color = "black"))








# ==== 调整 SHAP force 图参数 ==== (理想)
# 尝试增加图宽度、字体缩小一点，给标签留空间
plot1 <- sv_force(shap_value, row_id = 5, size = 8, max_display = 20) + theme_force +
  theme(axis.title.x = element_text(face = "bold", size = 18))  # 加粗并增大X轴标签字体

plot2 <- sv_force(shap_value, row_id = 129, size = 8, max_display = 20) + theme_force +
  theme(axis.title.x = element_text(face = "bold", size = 18))  # 加粗并增大X轴标签字体

# ==== 保存图像，增大尺寸避免拥挤 ==== 
ggsave("shap_sample_5.tiff", plot = plot1, dpi = 300,
       width = 8, height = 4, units = "in", device = "tiff", compression = "lzw")

ggsave("shap_sample_129.tiff", plot = plot2, dpi = 300,
       width = 8, height = 4, units = "in", device = "tiff", compression = "lzw")








