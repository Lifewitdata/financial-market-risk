# 01_inspect.R — Data-quality inspection
# Run from the project root:  source("R/01_inspect.R")

if (basename(getwd()) == "R") setwd("..")
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

raw <- read_csv("data/market_risk_daily.csv", col_types = cols(
  date = col_date(format = "%Y-%m-%d"),
  ticker = col_character(),
  open = col_double(), high = col_double(), low = col_double(),
  close = col_double(), volume = col_double(),
  daily_return = col_double(),
  realized_volatility_5d = col_double(),
  realized_volatility_20d = col_double(),
  high_volatility_next_5d = col_double()
), show_col_types = FALSE)

cat("== Shape ==\n")
cat("Rows:", nrow(raw), " Columns:", ncol(raw), "\n\n")

cat("== Variable types ==\n")
print(as.data.frame(sapply(raw, function(x) class(x)[1])))

cat("\n== Missing values ==\n")
miss <- data.frame(
  variable    = names(raw),
  n_missing   = as.integer(colSums(is.na(raw))),
  pct_missing = round(100 * colSums(is.na(raw)) / nrow(raw), 2),
  row.names   = NULL
)
print(miss)

cat("\n== Integrity checks ==\n")
cat("Duplicate rows:", sum(duplicated(raw)), "\n")
cat("Sorted chronologically:", all(diff(raw$date) > 0), "\n")
cat("Date range:", as.character(min(raw$date)), "to", as.character(max(raw$date)), "\n")
cat("Distinct tickers:", paste(unique(raw$ticker), collapse = ", "), "\n")
cat("OHLC violations (high < max(open,close) | low > min(open,close)):",
    sum(raw$high < pmax(raw$open, raw$close) | raw$low > pmin(raw$open, raw$close)), "\n")
cat("Non-positive volume rows:", sum(raw$volume <= 0), "\n")

cat("\n== Target balance (supplied practice label) ==\n")
tb <- raw %>%
  filter(!is.na(high_volatility_next_5d)) %>%
  summarise(n = n(),
            positives = sum(high_volatility_next_5d == 1),
            positive_rate = mean(high_volatility_next_5d == 1))
print(as.data.frame(tb))

cat("\n== Positive rate by calendar year ==\n")
yearly <- raw %>%
  filter(!is.na(high_volatility_next_5d)) %>%
  mutate(year = as.integer(format(date, "%Y"))) %>%
  group_by(year) %>%
  summarise(n = n(), positive_rate = round(mean(high_volatility_next_5d == 1), 4),
            .groups = "drop")
print(as.data.frame(yearly))

# Warm-up / tail NA positions (expected: trailing vol needs history,
# target needs a 5-day future window)
cat("\n== First non-NA row per trailing/missing column ==\n")
for (v in c("realized_volatility_5d", "realized_volatility_20d")) {
  idx <- which(!is.na(raw[[v]]))[1]
  cat(v, "-> first available:", as.character(raw$date[idx]), "\n")
}
cat("high_volatility_next_5d -> last available:", as.character(max(raw$date[!is.na(raw$high_volatility_next_5d)])), "\n")

write_csv(miss, "outputs/data_quality_missingness.csv")
write_csv(yearly, "outputs/target_rate_by_year.csv")
cat("\nSaved: outputs/data_quality_missingness.csv, outputs/target_rate_by_year.csv\n")
