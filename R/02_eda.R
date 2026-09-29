# 02_eda.R — Exploratory plots
# Run from the project root:  source("R/02_eda.R")

if (basename(getwd()) == "R") setwd("..")
source("R/utils.R")
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
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
), show_col_types = FALSE)

df <- df %>% mutate(target_lbl = factor(high_volatility_next_5d,
                                       levels = c(0, 1),
                                       labels = c("calm", "high-vol")))

# 1. Closing level with high-volatility 5-day windows marked
p1 <- ggplot(df, aes(date, close)) +
  geom_line(color = "grey30", linewidth = 0.5) +
  geom_point(data = filter(df, !is.na(target_lbl), target_lbl == "high-vol"),
             aes(date, close), color = "#D62728", size = 0.9, alpha = 0.6) +
  labs(title = "Index level with high-volatility 5-day windows (red)",
       x = NULL, y = "Close") +
  theme_mr()
ggsave("figures/eda/01_close_with_labels.png", p1, width = 10, height = 5, dpi = 150, bg = "white")

# 2. Daily returns
p2 <- ggplot(df, aes(date, daily_return)) +
  geom_line(color = "steelblue", linewidth = 0.4, alpha = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Daily log returns — volatility clusters over time",
       x = NULL, y = "Daily return") +
  theme_mr()
ggsave("figures/eda/02_daily_returns.png", p2, width = 10, height = 4.5, dpi = 150, bg = "white")

# 3. Trailing 20-day realized volatility with labels
p3 <- ggplot(df, aes(date, realized_volatility_20d)) +
  geom_line(color = "darkorange", linewidth = 0.6) +
  geom_point(data = filter(df, !is.na(target_lbl), target_lbl == "high-vol"),
             aes(date, realized_volatility_20d), color = "#D62728", size = 0.9, alpha = 0.6) +
  labs(title = "Trailing 20-day realized volatility (past-only) vs future high-vol labels",
       subtitle = "Past volatility is a feature; the label describes the *next* 5 days",
       x = NULL, y = "Trailing 20d vol") +
  theme_mr()
ggsave("figures/eda/03_trailing_volatility.png", p3, width = 10, height = 4.5, dpi = 150, bg = "white")

# 4. Volume
p4 <- ggplot(df, aes(date, volume)) +
  geom_line(color = "forestgreen", linewidth = 0.4, alpha = 0.8) +
  labs(title = "Daily volume", x = NULL, y = "Volume") +
  theme_mr()
ggsave("figures/eda/04_volume.png", p4, width = 10, height = 4.5, dpi = 150, bg = "white")

# 5. Target balance overall
tb <- df %>% filter(!is.na(target_lbl)) %>% count(target_lbl)
p5 <- ggplot(tb, aes(target_lbl, n, fill = target_lbl)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = n), vjust = -0.4) +
  scale_fill_manual(values = c(calm = "grey60", `high-vol` = "#D62728")) +
  labs(title = "Target balance (supplied practice label)",
       x = NULL, y = "Trading days") +
  theme_mr()
ggsave("figures/eda/05_target_balance.png", p5, width = 6, height = 4.5, dpi = 150, bg = "white")

# 6. Positive rate by year (regime check)
yr <- df %>% filter(!is.na(target_lbl)) %>%
  mutate(year = format(date, "%Y")) %>%
  group_by(year) %>% summarise(rate = mean(high_volatility_next_5d == 1), .groups = "drop")
p6 <- ggplot(yr, aes(year, rate)) +
  geom_col(fill = "steelblue") +
  geom_text(aes(label = sprintf("%.1f%%", 100 * rate)), vjust = -0.4, size = 3.5) +
  labs(title = "High-volatility rate by year — regimes shift",
       x = NULL, y = "Positive rate") +
  theme_mr()
ggsave("figures/eda/06_target_rate_by_year.png", p6, width = 8, height = 4.5, dpi = 150, bg = "white")

# 7. Return distribution by class
p7 <- ggplot(filter(df, !is.na(target_lbl), abs(daily_return) < 0.08),
             aes(daily_return, fill = target_lbl)) +
  geom_histogram(bins = 60, alpha = 0.6, position = "identity") +
  scale_fill_manual(values = c(calm = "grey60", `high-vol` = "#D62728")) +
  labs(title = "Return distribution by next-5-day label",
       subtitle = "Days that precede high-vol windows already show fatter tails",
       x = "Daily return (trimmed at +/-8%)", y = "Count", fill = NULL) +
  theme_mr()
ggsave("figures/eda/07_return_dist_by_class.png", p7, width = 8, height = 4.5, dpi = 150, bg = "white")

# 8. Past volatility (feature) vs future volatility (target basis)
# future 5d vol at t = trailing 5d sd ending at t+5 = lead(trail5, 5)
df <- df %>% mutate(
  trail5 = zoo::rollapplyr(daily_return, 5, sd, fill = NA),
  future_vol_5d = dplyr::lead(trail5, 5)
)
p8 <- ggplot(filter(df, !is.na(future_vol_5d), !is.na(target_lbl)),
             aes(realized_volatility_20d, future_vol_5d, color = target_lbl)) +
  geom_point(alpha = 0.35, size = 0.8) +
  scale_color_manual(values = c(calm = "grey60", `high-vol` = "#D62728")) +
  labs(title = "Past vs future volatility",
       subtitle = "Features use only data available at date t (x-axis); the target is built from t+1..t+5 (y-axis)",
       x = "Trailing 20d vol at t (feature)", y = "Realized vol over next 5d (target basis)",
       color = NULL) +
  theme_mr()
ggsave("figures/eda/08_past_vs_future_vol.png", p8, width = 8, height = 5, dpi = 150, bg = "white")

cat("EDA figures saved to figures/eda/\n")
