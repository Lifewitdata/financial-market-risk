# shiny/app.R — Market-risk dashboard
# Launch from the project root:  shiny::runApp("shiny")
# Expects the project layout (reads ../data, ../models, ../outputs).

suppressPackageStartupMessages({
  library(shiny)
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(ranger)  # needed for predict() on the fitted forest
})

P <- function(...) file.path("..", ...)

mt      <- NULL  # modeling table not needed at runtime; features come from the snapshot
meta    <- readRDS(P("models/meta.rds"))
glm_m   <- readRDS(P("models/glm_model.rds"))
rf_m    <- readRDS(P("models/rf_model.rds"))
base_m  <- readRDS(P("models/baseline.rds"))
preds   <- readRDS(P("outputs/test_predictions.rds"))
metrics <- read_csv(P("outputs/metrics_val_test.csv"), show_col_types = FALSE)
bt      <- read_csv(P("outputs/backtest_quarterly.csv"), show_col_types = FALSE)

features <- meta$features

# ---- current snapshot: latest trading date (features complete, target pending)
snap <- readRDS(P("data/latest_snapshot.rds"))
snap_date <- snap$date
scale_row <- function(row) {
  as.data.frame(mapply(function(x, m, s) (x - m) / s,
                       row[, features], glm_m$scaler$mu, glm_m$scaler$sd,
                       SIMPLIFY = FALSE))
}
p_glm_now <- as.numeric(predict(glm_m$fit, newdata = scale_row(snap), type = "response"))
p_rf_now  <- as.numeric(predict(rf_m$fit, data = snap[, features])$predictions[, "1"])
flag_now  <- ifelse(p_rf_now >= rf_m$threshold, "HIGH RISK", "calm")

risk_color <- function(flag) if (flag == "HIGH RISK") "#D62728" else "#2CA02C"

ui <- fluidPage(
  titlePanel("Market Risk Monitor — 5-day high-volatility forecast"),
  tabsetPanel(
    tabPanel("Current risk snapshot",
             br(),
             h4(paste("As of", snap_date, "(latest trading date; future window unavailable)")),
             fluidRow(
               column(4, wellPanel(
                 h5("Logistic regression"),
                 h3(sprintf("%.1f%%", 100 * p_glm_now)),
                 p(paste("Threshold:", sprintf("%.2f", glm_m$threshold)))
               )),
               column(4, wellPanel(
                 h5("Random forest"),
                 h3(sprintf("%.1f%%", 100 * p_rf_now)),
                 p(paste("Threshold:", sprintf("%.2f", rf_m$threshold)))
               )),
               column(4, wellPanel(
                 h5("Risk flag (random forest)"),
                 h3(style = paste0("color:", risk_color(flag_now)), flag_now),
                 p("Actual outcome: pending — the 5-day future window is unavailable, exactly as in production.")
               ))
             ),
             h4("Feature snapshot (past-only, available at this date)"),
             tableOutput("feat_table"),
             p(em("Features use only data available as of the snapshot date. The target describes the next 5 trading days."))
    ),
    tabPanel("Model performance",
             br(),
             h4("Test-period metrics (2024–2025; thresholds tuned on validation)"),
             tableOutput("metrics_table"),
             fluidRow(
               column(6, plotOutput("roc_plot")),
               column(6, plotOutput("pr_plot"))
             ),
             h4("Test confusion counts"),
             tableOutput("cm_table")
    ),
    tabPanel("Stability over time",
             br(),
             p("Expanding-window backtest: each quarter 2023-Q1 → 2025-Q4 scored by models refit on all prior data, with frozen thresholds."),
             plotOutput("bt_plot", height = "420px"),
             tableOutput("bt_table")
    ),
    tabPanel("About & limitations",
             br(),
             h4("What this is"),
             p("A market-risk classification exercise on ", strong("synthetic"), " data: predict whether the next 5 trading days are high-volatility, using only information available as of the current date."),
             h4("Limitations"),
             tags$ul(
               tags$li("Synthetic data — patterns do not transfer to real markets."),
               tags$li("Single series; no macro or cross-sectional features."),
               tags$li("The 75th-percentile 'high-vol' cutoff is defensible but arbitrary."),
               tags$li("False alarms vs misses need a real cost matrix from a risk desk."),
               tags$li("Performance drifts across regimes — a production system needs periodic refits and monitoring.")
             ),
             p(strong("Not investment advice. Not a trading recommendation."))
    )
  )
)

server <- function(input, output, session) {

  output$feat_table <- renderTable({
    data.frame(
      feature = features,
      value = round(as.numeric(snap[1, features]), 4),
      stringsAsFactors = FALSE
    )
  }, striped = TRUE, hover = TRUE)

  output$metrics_table <- renderTable({
    metrics %>% filter(split == "test") %>%
      mutate(across(c(threshold, precision, recall, f1, roc_auc, pr_auc), ~ round(.x, 3))) %>%
      select(model, threshold, precision, recall, f1, roc_auc, pr_auc)
  }, striped = TRUE, hover = TRUE)

  roc_points <- function(p, y) {
    r <- pROC::roc(y, p, quiet = TRUE)
    data.frame(fpr = 1 - rev(r$specificities), tpr = rev(r$sensitivities))
  }
  pr_points <- function(p, y) {
    pr <- PRROC::pr.curve(scores.class0 = p[y == 0], scores.class1 = p[y == 1], curve = TRUE)
    data.frame(recall = pr$curve[, 1], precision = pr$curve[, 2])
  }

  output$roc_plot <- renderPlot({
    df <- bind_rows(
      mutate(roc_points(preds$p_logistic, preds$y), model = "Logistic"),
      mutate(roc_points(preds$p_rf, preds$y), model = "Random forest"),
      mutate(roc_points(preds$p_baseline, preds$y), model = "Baseline rule")
    )
    ggplot(df, aes(fpr, tpr, color = model)) +
      geom_line(linewidth = 0.9) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
      coord_equal() + labs(title = "ROC — test", x = "FPR", y = "TPR", color = NULL) +
      theme_minimal()
  })

  output$pr_plot <- renderPlot({
    df <- bind_rows(
      mutate(pr_points(preds$p_logistic, preds$y), model = "Logistic"),
      mutate(pr_points(preds$p_rf, preds$y), model = "Random forest"),
      mutate(pr_points(preds$p_baseline, preds$y), model = "Baseline rule")
    )
    ggplot(df, aes(recall, precision, color = model)) +
      geom_line(linewidth = 0.9) +
      geom_hline(yintercept = mean(preds$y), linetype = "dashed", color = "grey50") +
      labs(title = "Precision–recall — test", x = "Recall", y = "Precision", color = NULL) +
      theme_minimal()
  })

  output$cm_table <- renderTable({
    cm <- read_csv(P("outputs/confusion_test.csv"), show_col_types = FALSE)
    cm %>% mutate(actual = ifelse(actual == 1, "high-vol", "calm"),
                  predicted = ifelse(predicted == 1, "high-vol", "calm")) %>%
      tidyr::pivot_wider(names_from = predicted, values_from = n, values_fill = 0)
  }, striped = TRUE)

  output$bt_plot <- renderPlot({
    ggplot(bt, aes(quarter, f1, color = model, group = model)) +
      geom_line(linewidth = 0.9) + geom_point(size = 2) +
      labs(title = "Quarterly F1 — expanding-window backtest",
           x = NULL, y = "F1", color = NULL) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })

  output$bt_table <- renderTable({
    bt %>% mutate(across(c(precision, recall, f1, roc_auc, pr_auc), ~ round(.x, 3)))
  }, striped = TRUE, hover = TRUE)
}

shinyApp(ui, server)
