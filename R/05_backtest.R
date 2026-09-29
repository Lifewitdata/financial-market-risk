# 05_backtest.R — Expanding-window quarterly backtest (model stability over time)
# Run from the project root:  source("R/05_backtest.R")
#
# For each calendar quarter from 2023-Q1 to 2025-Q4: refit both models on ALL
# data strictly before the quarter (expanding window), then score the quarter.
# The high-vol threshold stays FIXED at its training-period value, and the
# operating thresholds stay FIXED at their validation-tuned values — this
# diagnostic measures stability, not a re-tuned generalization estimate.

if (basename(getwd()) == "R") setwd("..")
source("R/utils.R")
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(ranger)
})

set.seed(42)
mt   <- readRDS("data/modeling_table.rds")
meta <- readRDS("models/meta.rds")
features <- meta$features
thr_glm <- readRDS("models/glm_model.rds")$threshold
thr_rf  <- readRDS("models/rf_model.rds")$threshold

quarters <- seq(as.Date("2023-01-01"), as.Date("2025-10-01"), by = "quarter")
starts <- quarters
ends   <- c(quarters[-1] - 1, as.Date("2025-12-31"))

fit_glm <- function(d) {
  mu <- vapply(d[, features], mean, numeric(1))
  sdv <- vapply(d[, features], sd, numeric(1))
  Xs <- as.data.frame(mapply(function(x, m, s) (x - m) / s, d[, features], mu, sdv,
                             SIMPLIFY = FALSE))
  list(fit = glm(y ~ ., data = data.frame(y = d$target, Xs), family = binomial()),
       mu = mu, sdv = sdv)
}
pred_glm <- function(m, d) {
  Xs <- as.data.frame(mapply(function(x, mm, s) (x - mm) / s, d[, features], m$mu, m$sdv,
                             SIMPLIFY = FALSE))
  as.numeric(predict(m$fit, newdata = Xs, type = "response"))
}

results <- list()
for (i in seq_along(starts)) {
  qtr <- paste0(format(starts[i], "%Y"), "-Q", (as.integer(format(starts[i], "%m")) - 1) %/% 3 + 1)
  fit_d <- mt %>% filter(date < starts[i])
  tst_d <- mt %>% filter(date >= starts[i], date <= ends[i])
  if (nrow(tst_d) < 20 || sum(fit_d$target) < 10 || sum(fit_d$target == 0) < 10 ||
      sum(tst_d$target) < 3 || sum(tst_d$target == 0) < 3) next

  g <- fit_glm(fit_d)
  r <- ranger(y ~ ., data = data.frame(y = factor(fit_d$target), fit_d[, features]),
              probability = TRUE, num.trees = 300,
              mtry = max(1, floor(sqrt(length(features)))),
              min.node.size = 20, seed = 42)

  pg <- pred_glm(g, tst_d)
  pr <- predict(r, data = tst_d[, features])$predictions[, "1"]
  mg <- classification_metrics(pg, tst_d$target, thr_glm)
  mr <- classification_metrics(pr, tst_d$target, thr_rf)

  results[[i]] <- bind_rows(
    mutate(mg, model = "logistic"), mutate(mr, model = "random_forest")
  ) %>% mutate(quarter = qtr) %>%
    select(quarter, model, n, precision, recall, f1, roc_auc, pr_auc)
  cat(sprintf("%s  fit n=%d  test n=%d  glm F1=%.3f  rf F1=%.3f\n",
              qtr, nrow(fit_d), nrow(tst_d), mg$f1, mr$f1))
}

bt <- bind_rows(results)
write_csv(bt, "outputs/backtest_quarterly.csv")

p <- ggplot(bt, aes(quarter, f1, color = model, group = model)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  labs(title = "Expanding-window backtest — F1 per quarter",
       subtitle = "Models refit on all prior data; thresholds frozen from validation",
       x = NULL, y = "F1", color = NULL) +
  theme_mr() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("figures/model/backtest_f1_quarterly.png", p, width = 10, height = 5, dpi = 150, bg = "white")

p2 <- ggplot(bt, aes(quarter, recall, color = model, group = model)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  labs(title = "Expanding-window backtest — recall per quarter",
       subtitle = "Recall = share of high-vol windows caught",
       x = NULL, y = "Recall", color = NULL) +
  theme_mr() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("figures/model/backtest_recall_quarterly.png", p2, width = 10, height = 5, dpi = 150, bg = "white")

cat("\nSaved: outputs/backtest_quarterly.csv, figures/model/backtest_*.png\n")
cat(sprintf("Mean quarterly F1 — logistic: %.3f | random forest: %.3f\n",
            mean(bt$f1[bt$model == "logistic"]), mean(bt$f1[bt$model == "random_forest"])))
cat(sprintf("SD quarterly F1   — logistic: %.3f | random forest: %.3f\n",
            sd(bt$f1[bt$model == "logistic"]), sd(bt$f1[bt$model == "random_forest"])))
