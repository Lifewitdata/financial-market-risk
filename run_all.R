# run_all.R — End-to-end pipeline. Run from the project root:
#   source("run_all.R")
# Step 6 (report) renders the R Markdown report to report/market_risk_report.html.

message("== [1/6] Data inspection ==")
source("R/01_inspect.R")

message("\n== [2/6] Exploratory analysis ==")
source("R/02_eda.R")

message("\n== [3/6] Target definition + feature engineering ==")
source("R/03_features_target.R")

message("\n== [4/6] Model training + evaluation ==")
source("R/04_train_evaluate.R")

message("\n== [5/6] Expanding-window backtest ==")
source("R/05_backtest.R")

message("\n== [6/6] Rendering report ==")
rmarkdown::render("report/market_risk_report.Rmd",
                  output_file = "market_risk_report.html",
                  output_dir = "report", quiet = TRUE)

message("\nDone. Open report/market_risk_report.html to read the report.")
