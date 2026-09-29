# 04_train_evaluate.R — Train logistic regression vs random forest, tune threshold
# on validation, evaluate on the held-out test period.
# Run from the project root:  source("R/04_train_evaluate.R")

if (basename(getwd()) == "R") setwd("..")
source("R/utils.R")
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(tidyr)
  library(ranger)
  library(pROC)
  library(PRROC)
})

set.seed(42)
mt   <- readRDS("data/modeling_table.rds")
meta <- readRDS("models/meta.rds")
features <- meta$features

train <- mt %>% filter(split == "train")
val   <- mt %>% filter(split == "validation")
test  <- mt %>% filter(split == "test")
cat(sprintf("Train n=%d | Validation n=%d | Test n=%d\n", nrow(train), nrow(val), nrow(test)))

# ------------------------------------------------- logistic regression (scaled)
sc <- list(mu = vapply(train[, features], mean, numeric(1)),
           sd = vapply(train[, features], sd, numeric(1)))
scale_X <- function(d) {
  as.data.frame(mapply(function(x, m, s) (x - m) / s,
                        d[, features], sc$mu, sc$sd, SIMPLIFY = FALSE))
}
glm_fit <- glm(y ~ ., data = data.frame(y = train$target, scale_X(train)),
               family = binomial())
p_val_glm  <- as.numeric(predict(glm_fit, newdata = scale_X(val), type = "response"))
p_test_glm <- as.numeric(predict(glm_fit, newdata = scale_X(test), type = "response"))

# ---------------------------------------------------------------- random forest
rf_fit <- ranger(y ~ .,
                 data = data.frame(y = factor(train$target), train[, features]),
                 probability = TRUE, num.trees = 500,
                 mtry = max(1, floor(sqrt(length(features)))),
                 min.node.size = 20, importance = "impurity", seed = 42)
p_val_rf  <- predict(rf_fit, data = val[, features])$predictions[, "1"]
p_test_rf <- predict(rf_fit, data = test[, features])$predictions[, "1"]

# ------------------------------------------------- simple rule baseline (past-only)
# Predict high-vol when trailing 20d vol exceeds its training-period 75th pct.
rv20_q75 <- meta$rv20_train_q75
p_val_base  <- val$realized_volatility_20d    # score for ROC/PR curves
p_test_base <- test$realized_volatility_20d
pred_val_base  <- as.integer(val$realized_volatility_20d >= rv20_q75)
pred_test_base <- as.integer(test$realized_volatility_20d >= rv20_q75)

# ------------------------------------------------- threshold tuning (validation)
thr_glm <- best_threshold_f1(p_val_glm, val$target)
thr_rf  <- best_threshold_f1(p_val_rf, val$target)
cat(sprintf("\nChosen thresholds (max F1 on validation): logistic=%.2f  rf=%.2f\n",
            thr_glm, thr_rf))

# ------------------------------------------------------------------ evaluate
m_val_glm  <- classification_metrics(p_val_glm, val$target, thr_glm)
m_val_rf   <- classification_metrics(p_val_rf, val$target, thr_rf)
m_test_glm <- classification_metrics(p_test_glm, test$target, thr_glm)
m_test_rf  <- classification_metrics(p_test_rf, test$target, thr_rf)
# baseline metrics at its fixed operating point
base_metrics <- function(y, pred, score) {
  tp <- sum(y == 1 & pred == 1); fp <- sum(y == 0 & pred == 1)
  fn <- sum(y == 1 & pred == 0)
  prec <- tp / (tp + fp); rec <- tp / (tp + fn)
  data.frame(threshold = NA_real_, n = length(y),
             precision = prec, recall = rec,
             f1 = 2 * prec * rec / (prec + rec),
             roc_auc = as.numeric(pROC::auc(pROC::roc(y, score, quiet = TRUE))),
             pr_auc = pr_auc(score, y),
             tp = tp, fp = fp, fn = fn, tn = sum(y == 0 & pred == 0))
}
m_val_base  <- base_metrics(val$target, pred_val_base, p_val_base)
m_test_base <- base_metrics(test$target, pred_test_base, p_test_base)

metrics <- bind_rows(
  mutate(m_val_glm, model = "logistic", split = "validation"),
  mutate(m_val_rf, model = "random_forest", split = "validation"),
  mutate(m_val_base, model = "baseline_rule", split = "validation"),
  mutate(m_test_glm, model = "logistic", split = "test"),
  mutate(m_test_rf, model = "random_forest", split = "test"),
  mutate(m_test_base, model = "baseline_rule", split = "test")
) %>% select(model, split, threshold, n, precision, recall, f1, roc_auc, pr_auc,
             tp, fp, fn, tn)
print(as.data.frame(metrics %>% select(model, split, threshold, precision, recall, f1, roc_auc, pr_auc)))
write_csv(metrics, "outputs/metrics_val_test.csv")

# Confusion matrices on test
cm <- bind_rows(
  data.frame(model = "logistic",      actual = test$target, predicted = as.integer(p_test_glm >= thr_glm)),
  data.frame(model = "random_forest", actual = test$target, predicted = as.integer(p_test_rf >= thr_rf)),
  data.frame(model = "baseline_rule", actual = test$target, predicted = pred_test_base)
)
write_csv(cm %>% count(model, actual, predicted, name = "n"), "outputs/confusion_test.csv")

# Save test predictions for the report / Shiny app
saveRDS(data.frame(date = test$date, y = test$target,
                   p_logistic = p_test_glm, p_rf = p_test_rf, p_baseline = p_test_base),
        "outputs/test_predictions.rds")

# ------------------------------------------------------------- save models
saveRDS(list(fit = glm_fit, scaler = sc, threshold = thr_glm, features = features),
        "models/glm_model.rds")
saveRDS(list(fit = rf_fit, threshold = thr_rf, features = features),
        "models/rf_model.rds")
saveRDS(list(rv20_q75 = rv20_q75), "models/baseline.rds")
cat("\nSaved models to models/ and metrics to outputs/\n")

# ------------------------------------------------------------------- figures
roc_df <- bind_rows(
  data.frame(model = "Logistic", fpr = 1 - rev(pROC::roc(test$target, p_test_glm, quiet = TRUE)$specificities),
             tpr = rev(pROC::roc(test$target, p_test_glm, quiet = TRUE)$sensitivities)),
  data.frame(model = "Random forest", fpr = 1 - rev(pROC::roc(test$target, p_test_rf, quiet = TRUE)$specificities),
             tpr = rev(pROC::roc(test$target, p_test_rf, quiet = TRUE)$sensitivities)),
  data.frame(model = "Baseline rule", fpr = 1 - rev(pROC::roc(test$target, p_test_base, quiet = TRUE)$specificities),
             tpr = rev(pROC::roc(test$target, p_test_base, quiet = TRUE)$sensitivities))
)
p_roc <- ggplot(roc_df, aes(fpr, tpr, color = model)) +
  geom_line(linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  coord_equal() +
  labs(title = "ROC curves — test period (2024–2025)",
       x = "False positive rate", y = "True positive rate", color = NULL) +
  theme_mr()
ggsave("figures/model/roc_test.png", p_roc, width = 7, height = 6, dpi = 150, bg = "white")

pr_curve <- function(p, y) {
  pr <- PRROC::pr.curve(scores.class0 = p[y == 0], scores.class1 = p[y == 1], curve = TRUE)
  data.frame(recall = pr$curve[, 1], precision = pr$curve[, 2])
}
pr_df <- bind_rows(
  mutate(pr_curve(p_test_glm, test$target), model = "Logistic"),
  mutate(pr_curve(p_test_rf, test$target), model = "Random forest"),
  mutate(pr_curve(p_test_base, test$target), model = "Baseline rule")
)
base_rate <- mean(test$target)
p_pr <- ggplot(pr_df, aes(recall, precision, color = model)) +
  geom_line(linewidth = 0.9) +
  geom_hline(yintercept = base_rate, linetype = "dashed", color = "grey50") +
  labs(title = "Precision–recall curves — test period (2024–2025)",
       subtitle = sprintf("Dashed line = test positive rate (%.1f%%)", 100 * base_rate),
       x = "Recall", y = "Precision", color = NULL) +
  theme_mr()
ggsave("figures/model/pr_test.png", p_pr, width = 7, height = 6, dpi = 150, bg = "white")

# Confusion matrix heatmap (random forest, test)
cm_rf <- as.data.frame(table(Actual = factor(test$target, levels = c(0, 1), labels = c("calm", "high-vol")),
                             Predicted = factor(as.integer(p_test_rf >= thr_rf),
                                                levels = c(0, 1), labels = c("calm", "high-vol"))))
p_cm <- ggplot(cm_rf, aes(Predicted, Actual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 7) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion matrix — random forest, test period",
       subtitle = sprintf("Threshold %.2f (tuned on validation)", thr_rf)) +
  theme_mr() + theme(legend.position = "none")
ggsave("figures/model/confusion_rf_test.png", p_cm, width = 6, height = 5, dpi = 150, bg = "white")

# Random forest variable importance
imp <- data.frame(feature = names(rf_fit$variable.importance),
                  importance = as.numeric(rf_fit$variable.importance)) %>%
  arrange(desc(importance))
p_imp <- ggplot(imp, aes(reorder(feature, importance), importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Random forest — variable importance (impurity)",
       x = NULL, y = "Importance") +
  theme_mr()
ggsave("figures/model/rf_importance.png", p_imp, width = 8, height = 5.5, dpi = 150, bg = "white")
write_csv(imp, "outputs/rf_importance.csv")

# Logistic regression odds ratios
coefs <- as.data.frame(summary(glm_fit)$coefficients)
coefs$feature <- rownames(coefs)
coefs <- coefs %>% filter(feature != "(Intercept)") %>%
  mutate(or = exp(Estimate),
         lo = exp(Estimate - 1.96 * `Std. Error`),
         hi = exp(Estimate + 1.96 * `Std. Error`)) %>%
  arrange(or)
p_or <- ggplot(coefs, aes(reorder(feature, or), or)) +
  geom_point(size = 2.5, color = "#D62728") +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.2, color = "#D62728") +
  geom_hline(yintercept = 1, linetype = "dashed") +
  coord_flip() +
  labs(title = "Logistic regression — odds ratios (per 1 SD of feature)",
       x = NULL, y = "Odds ratio (95% CI)") +
  theme_mr()
ggsave("figures/model/glm_odds_ratios.png", p_or, width = 8, height = 5.5, dpi = 150, bg = "white")

cat("Model figures saved to figures/model/\n")
