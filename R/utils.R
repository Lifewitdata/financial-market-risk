# R/utils.R — shared helper functions (no side effects; source from other scripts)

f1_score <- function(y_true, y_pred) {
  tp <- sum(y_true == 1 & y_pred == 1)
  fp <- sum(y_true == 0 & y_pred == 1)
  fn <- sum(y_true == 1 & y_pred == 0)
  if ((tp + fp) == 0 || (tp + fn) == 0) return(0)
  prec <- tp / (tp + fp)
  rec  <- tp / (tp + fn)
  if ((prec + rec) == 0) return(0)
  2 * prec * rec / (prec + rec)
}

# Pick the probability threshold that maximises F1 on a labelled set
best_threshold_f1 <- function(p, y, grid = seq(0.05, 0.95, by = 0.01)) {
  f1s <- vapply(grid, function(t) f1_score(y, as.integer(p >= t)), numeric(1))
  grid[which.max(f1s)]
}

# Area under the precision-recall curve (PRROC integral)
pr_auc <- function(p, y) {
  PRROC::pr.curve(scores.class0 = p[y == 0],
                  scores.class1 = p[y == 1],
                  curve = FALSE)$auc.integral
}

# Full classification report at a fixed threshold
classification_metrics <- function(p, y, threshold = 0.5) {
  pred <- as.integer(p >= threshold)
  tp <- sum(y == 1 & pred == 1); fp <- sum(y == 0 & pred == 1)
  fn <- sum(y == 1 & pred == 0); tn <- sum(y == 0 & pred == 0)
  prec <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  rec  <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  f1   <- if (!is.na(prec) && !is.na(rec) && (prec + rec) > 0) {
    2 * prec * rec / (prec + rec)
  } else 0
  roc_auc <- as.numeric(pROC::auc(pROC::roc(y, p, quiet = TRUE)))
  data.frame(threshold = threshold, n = length(y),
             precision = prec, recall = rec, f1 = f1,
             roc_auc = roc_auc, pr_auc = pr_auc(p, y),
             tp = tp, fp = fp, fn = fn, tn = tn)
}

theme_mr <- function() {
  ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(plot.title = ggplot2::element_text(face = "bold"),
                   legend.position = "bottom",
                   # explicit white canvas: some R builds default the png
                   # device to a dark background when plot.background is blank
                   plot.background = ggplot2::element_rect(fill = "white", colour = NA),
                   panel.background = ggplot2::element_rect(fill = "white", colour = NA))
}
