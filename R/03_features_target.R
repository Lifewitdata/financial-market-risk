# 03_features_target.R — Rigorous target definition + past-only feature engineering
# Run from the project root:  source("R/03_features_target.R")
#
# Leakage discipline:
#   * Features use ONLY information available as of date t (lags, trailing windows).
#   * The supplied label `high_volatility_next_5d` is a practice label — it is NOT a predictor.
#   * `future_vol_5d` (realized vol over t+1..t+5) is the target basis — never a predictor.
#   * The high-vol threshold is estimated on the TRAINING period only, then frozen
#     and applied to validation/test.

if (basename(getwd()) == "R") setwd("..")
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(zoo)
})

df <- read_csv("data/market_risk_daily.csv", col_types = cols(
  date = col_date(format = "%Y-%m-%d"),
  ticker = col_character(),
  open = col_double(), high = col_double(), low = col_double(),
  close = col_double(), volume = col_double(),
  daily_return = col_double(),
  realized_volatility_5d = col_double(),
  realized_volatility_20d = col_double(),
  high_volatility_next_5d = col_double()
), show_col_types = FALSE) %>% arrange(date)

TRAIN_END <- as.Date("2022-12-31")
VAL_END   <- as.Date("2023-12-31")

# ---------------------------------------------------------------- target ---
# Future 5-day realized volatility: sd of daily returns over t+1..t+5.
# trail5[t] covers [t-4, t], so shifting it forward 5 steps gives [t+1, t+5].
# (lead(., 1) would be wrong — it would mix 4 past returns with 1 future one.)
trail5 <- zoo::rollapplyr(df$daily_return, 5, sd, fill = NA)
df$future_vol_5d <- dplyr::lead(trail5, 5)

train_mask <- df$date <= TRAIN_END
# Threshold = 75th percentile of future 5d vol ON TRAINING DATA ONLY
vol_threshold <- as.numeric(quantile(df$future_vol_5d[train_mask], 0.75, na.rm = TRUE))
df$target <- as.integer(df$future_vol_5d >= vol_threshold)

cat("== Rigorous target definition ==\n")
cat("Basis: sd of daily returns over t+1..t+5 (future realized volatility)\n")
cat("Threshold: 75th percentile on TRAINING period only (",
    as.character(min(df$date[train_mask])), " to ", as.character(TRAIN_END), ")\n", sep = "")
cat(sprintf("Frozen threshold applied to validation/test: %.6f\n", vol_threshold))

# Agreement with the supplied practice label (where both are defined)
cmp <- df %>% filter(!is.na(target), !is.na(high_volatility_next_5d))
agree <- mean(cmp$target == cmp$high_volatility_next_5d)
cat(sprintf("\nAgreement rigorous target vs supplied label: %.1f%% (n=%d)\n",
            100 * agree, nrow(cmp)))
cat("Positive rate — rigorous:", round(mean(df$target, na.rm = TRUE), 4),
    " supplied:", round(mean(df$high_volatility_next_5d, na.rm = TRUE), 4), "\n")
write_csv(data.frame(threshold = vol_threshold,
                     train_start = as.character(min(df$date[train_mask])),
                     train_end = as.character(TRAIN_END),
                     agreement_with_supplied = round(agree, 4),
                     positive_rate_rigorous = round(mean(df$target, na.rm = TRUE), 4)),
          "outputs/target_definition.csv")

# --------------------------------- verify supplied rv columns are trailing ---
# The CSV ships realized_volatility_5d/20d. If they are trailing (past-only)
# they are legitimate features; if they peeked into the future they must be dropped.
chk <- df %>%
  mutate(rv5_recalc  = zoo::rollapplyr(daily_return, 5, sd, fill = NA),
         rv20_recalc = zoo::rollapplyr(daily_return, 20, sd, fill = NA)) %>%
  filter(!is.na(realized_volatility_5d), !is.na(rv5_recalc),
         !is.na(realized_volatility_20d), !is.na(rv20_recalc))
d5  <- max(abs(chk$realized_volatility_5d - chk$rv5_recalc))
d20 <- max(abs(chk$realized_volatility_20d - chk$rv20_recalc))
cat("\n== Supplied trailing-vol columns: leakage check ==\n")
cat(sprintf("max |supplied - recomputed trailing|  5d: %.2e   20d: %.2e\n", d5, d20))
cat("correlations:", round(cor(chk$realized_volatility_5d, chk$rv5_recalc), 6),
    round(cor(chk$realized_volatility_20d, chk$rv20_recalc), 6), "\n")
# Tolerance 1e-5: the CSV stores these columns rounded to 6 decimals, so a
# sub-microsecond-level difference is expected rounding noise, not leakage.
# (A look-ahead leak would show large, systematic differences.)
stopifnot(d5 < 1e-5, d20 < 1e-5)
cat("VERIFIED: both columns are trailing (past-only) — safe to use as features.\n")

# ------------------------------------------------------- feature engineering ---
rsi_wilder <- function(price, n = 14) {
  delta <- c(NA_real_, diff(price))
  gain  <- ifelse(is.na(delta), NA_real_, pmax(delta, 0))
  loss  <- ifelse(is.na(delta), NA_real_, pmax(-delta, 0))
  avg_gain <- avg_loss <- rep(NA_real_, length(price))
  avg_gain[n + 1] <- mean(gain[2:(n + 1)], na.rm = TRUE)
  avg_loss[n + 1] <- mean(loss[2:(n + 1)], na.rm = TRUE)
  for (i in (n + 2):length(price)) {
    avg_gain[i] <- (avg_gain[i - 1] * (n - 1) + gain[i]) / n
    avg_loss[i] <- (avg_loss[i - 1] * (n - 1) + loss[i]) / n
  }
  rs  <- avg_gain / avg_loss
  rsi <- 100 - 100 / (1 + rs)
  rsi[avg_loss == 0 & !is.na(avg_loss)] <- 100
  rsi[is.nan(rsi)] <- 50
  rsi
}

df <- df %>% mutate(
  # lagged returns (past only)
  ret_lag1 = lag(daily_return, 1),
  ret_lag2 = lag(daily_return, 2),
  ret_lag3 = lag(daily_return, 3),
  ret_lag4 = lag(daily_return, 4),
  ret_lag5 = lag(daily_return, 5),
  # trailing return moments
  mean_ret_5  = zoo::rollapplyr(daily_return, 5, mean, fill = NA),
  mean_ret_20 = zoo::rollapplyr(daily_return, 20, mean, fill = NA),
  sd_ret_10   = zoo::rollapplyr(daily_return, 10, sd, fill = NA),
  # volume regime (past only)
  log_volume = log(volume),
  vol_ma20   = zoo::rollapplyr(volume, 20, mean, fill = NA),
  vol_ratio  = volume / vol_ma20,
  # price structure (past only)
  range_5d       = zoo::rollapplyr((high - low) / close, 5, mean, fill = NA),
  price_vs_ma20  = close / zoo::rollapplyr(close, 20, mean, fill = NA) - 1,
  drawdown_60    = close / zoo::rollapplyr(close, 60, max, fill = NA) - 1,
  rsi14          = rsi_wilder(close, 14)
)

features <- c("ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5",
              "mean_ret_5", "mean_ret_20", "sd_ret_10",
              "realized_volatility_5d", "realized_volatility_20d",
              "log_volume", "vol_ratio", "range_5d",
              "price_vs_ma20", "drawdown_60", "rsi14")

df <- df %>% mutate(
  split = case_when(
    date <= TRAIN_END ~ "train",
    date <= VAL_END   ~ "validation",
    TRUE              ~ "test"
  )
)

# Analysis-ready table: complete features + defined target only
modeling <- df %>% filter(if_all(all_of(features), ~ !is.na(.x)), !is.na(target))

cat("\n== Modeling table ==\n")
cat("Features:", length(features), "\n")
print(as.data.frame(table(modeling$split)))
cat("Rows dropped (warm-up NA features or unavailable future window):",
    nrow(df) - nrow(modeling), "\n")
cat("Positive rate by split (rigorous target):\n")
print(as.data.frame(modeling %>% group_by(split) %>%
                      summarise(n = n(), positive_rate = round(mean(target), 4),
                                .groups = "drop")))

saveRDS(modeling, "data/modeling_table.rds")
write_csv(modeling %>% select(date, all_of(features), split, target,
                              future_vol_5d, high_volatility_next_5d),
          "data/modeling_table.csv")

# Latest-date feature snapshot for the Shiny "current risk" panel.
# Uses the final trading date (features complete; future window unavailable,
# so no target — exactly the production situation).
latest <- df %>%
  filter(if_all(all_of(features), ~ !is.na(.x))) %>%
  arrange(date) %>%
  slice(n()) %>%
  select(date, all_of(features))
saveRDS(latest, "data/latest_snapshot.rds")
cat("Latest snapshot date:", as.character(latest$date), "\n")
saveRDS(list(features = features,
             vol_threshold = vol_threshold,
             train_end = TRAIN_END, val_end = VAL_END,
             rv20_train_q75 = as.numeric(quantile(
               modeling$realized_volatility_20d[modeling$split == "train"], 0.75))),
        "models/meta.rds")
cat("\nSaved: data/modeling_table.rds/.csv, models/meta.rds\n")
