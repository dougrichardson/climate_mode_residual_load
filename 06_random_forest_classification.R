rm(list = ls())
gc()

library(dplyr)
library(tidyr)
library(ranger)


# Functions -------------------------------------------------------------------

# Read data for a given period and state.
Read_data <- function(folder, period = "historical", state = "NSW") {
  filename <- if (period == "historical") {
    file.path(
      folder,
      paste0(
        "current_monthly_energy_climate_modes_",
        state,
        "_detrended_anomalies.csv"
      )
    )
  } else {
    file.path(
      folder,
      paste0(
        "SC2050_monthly_energy_climate_modes_",
        state,
        "_detrended_anomalies.csv"
      )
    )
  }

  data <- read.csv(filename)
  return(data)
}


# Bin a continuous target into quantile-based classes.
bin_target_to_classes <- function(data, target, training_years, testing_years, n_bins) {
  training_data <- subset(data, Year %in% training_years)

  probs <- seq(0, 1, length.out = n_bins + 1)
  bins <- quantile(training_data[[target]], probs = probs, na.rm = TRUE)
  bins[1] <- -Inf
  bins[length(bins)] <- Inf

  data[[target]] <- cut(
    data[[target]],
    breaks = bins,
    labels = 1:n_bins,
    include.lowest = TRUE,
    right = TRUE
  )
  data[[target]] <- as.integer(as.character(data[[target]]))

  return(list(
    data_class = data,
    bins = bins
  ))
}


# Train a random forest classification model.
train_RF <- function(data, target, predictors, training_years, testing_years, seed = 42) {
  training_data <- subset(data, Year %in% training_years)
  testing_data <- subset(data, Year %in% testing_years)

  if (nrow(testing_data) == 0) {
    stop("Testing data is empty. Check your Year filtering or dataset.")
  }

  training_data[[target]] <- as.factor(training_data[[target]])
  testing_data[[target]] <- as.factor(testing_data[[target]])

  set.seed(seed)
  formula <- as.formula(paste(target, "~", paste(predictors, collapse = "+")))

  rf_model <- ranger(
    formula = formula,
    data = training_data,
    num.trees = 500,
    importance = "impurity",
    seed = seed,
    probability = FALSE
  )

  predictions <- predict(rf_model, data = testing_data)$predictions
  if (is.null(predictions)) {
    stop("Predictions failed. Check the model or testing data.")
  }

  testing_data$Prediction <- predictions
  training_data$Prediction <- NA
  combined_data <- rbind(training_data, testing_data)

  return(combined_data)
}


# Compute multiclass classification metrics.
compute_multiclass_metrics <- function(data, target, prediction_column, testing_years, all_classes = NULL) {
  testing_data <- subset(data, Year %in% testing_years)
  if (nrow(testing_data) == 0) {
    stop("Testing data is empty.")
  }

  actual <- testing_data[[target]]
  predicted <- testing_data[[prediction_column]]
  if (any(is.na(predicted))) {
    stop("Predicted values contain NAs.")
  }

  if (is.null(all_classes)) {
    all_classes <- sort(unique(c(actual, predicted)))
  }

  actual <- factor(actual, levels = all_classes)
  predicted <- factor(predicted, levels = all_classes)

  cm <- table(Predicted = predicted, Actual = actual)

  recall <- rep(NA, length(all_classes))
  precision <- rep(NA, length(all_classes))
  f1 <- rep(NA, length(all_classes))
  support <- rep(0, length(all_classes))

  for (i in seq_along(all_classes)) {
    class <- all_classes[i]
    TP <- cm[class, class]
    FN <- sum(cm[, class]) - TP
    FP <- sum(cm[class, ]) - TP

    support[i] <- sum(cm[, class])
    recall[i] <- if ((TP + FN) > 0) TP / (TP + FN) else NA
    precision[i] <- if ((TP + FP) > 0) TP / (TP + FP) else NA
    f1[i] <- if (!is.na(precision[i]) && !is.na(recall[i]) &&
      (precision[i] + recall[i]) > 0) {
      2 * precision[i] * recall[i] / (precision[i] + recall[i])
    } else {
      NA
    }
  }

  macro_recall <- mean(recall, na.rm = TRUE)
  macro_precision <- mean(precision, na.rm = TRUE)
  macro_f1 <- mean(f1, na.rm = TRUE)
  accuracy <- sum(diag(cm)) / sum(cm)

  metrics <- list(
    Accuracy = accuracy,
    Macro_Precision = macro_precision,
    Macro_Recall = macro_recall,
    Macro_F1 = macro_f1,
    Recall_per_class = setNames(recall, all_classes),
    Precision_per_class = setNames(precision, all_classes),
    F1_per_class = setNames(f1, all_classes),
    Support_per_class = setNames(support, all_classes),
    Confusion_Matrix = cm
  )

  return(metrics)
}


# Settings --------------------------------------------------------------------

folder <- "/g/data/w42/dr6273/work/seasonal_energy/"
res_path <- "/g/data/w42/dr6273/work/projects/Aus_energy/monthly_classification/results_8Jan26/"

n_bins <- 3
training_years <- 1940:2000
testing_years <- 2001:2023

targets <- c("Demand", "Wind", "Solar", "Residual_load")
states <- c("TAS", "VIC", "NSW", "NEM", "QLD", "SA")
seasons <- c("DJF", "MAM", "JJA", "SON")
# seasons <- c("ASO")

MonthsList <- list(
  "DJF" = c(12, 1, 2),
  "MAM" = c(3, 4, 5),
  "JJA" = c(6, 7, 8),
  "SON" = c(9, 10, 11),
  "ASO" = c(8, 9, 10)
)

predictors <- c("Nino34", "DMI", "SAM", "t2m")

all_combinations <- lapply(seq_along(predictors), function(k) {
  combn(predictors, k, simplify = FALSE)
})
all_combinations <- unlist(all_combinations, recursive = FALSE)

df <- data.frame(
  experiment = seq_along(all_combinations),
  predictors = sapply(all_combinations, paste, collapse = " "),
  stringsAsFactors = FALSE
)



# Run models ------------------------------------------------------------------

for (period in c("historical", "future")) {
  for (season in seasons) {
    months <- MonthsList[[season]]
    print(paste("Season:", season, "- Months:", paste(months, collapse = ", ")))

    for (state in states) {
      print(paste("Processing state:", state))
      results_df <- data.frame()

      for (target in targets) {
        print(paste("Target variable:", target))

        for (k in seq_along(all_combinations)) {
          selected_predictors <- all_combinations[[k]]

          for (seed in 1:30) {
            data_state_detrended <- Read_data(folder, period, state = state)
            data_state_detrended <- na.omit(data_state_detrended)
            data_state_detrended <- subset(data_state_detrended, Month %in% months)

            error_occurred <- FALSE
            results_binning <- tryCatch({
              bin_target_to_classes(
                data_state_detrended,
                target,
                training_years,
                testing_years,
                n_bins
              )
            }, error = function(e) {
              error_occurred <<- TRUE
              NULL
            })

            if (error_occurred || is.null(results_binning)) next

            data_state_detrended_binned <- results_binning[[1]]

            result_data <- train_RF(
              data = data_state_detrended_binned,
              target = target,
              predictors = c("Month", selected_predictors),
              training_years = training_years,
              testing_years = testing_years,
              seed = seed
            )

            metrics <- compute_multiclass_metrics(
              data = result_data,
              target = target,
              prediction_column = "Prediction",
              testing_years = testing_years,
              all_classes = 1:n_bins
            )

            results_df <- rbind(results_df, data.frame(
              n_bins = n_bins,
              experiment = k,
              Target = target,
              seed = seed,
              Accuracy = metrics$Accuracy,
              Macro_Precision = metrics$Macro_Precision,
              Macro_Recall = metrics$Macro_Recall,
              Macro_F1 = metrics$Macro_F1
            ))
          }
        }
      }

      write.csv(
        results_df,
        paste0(
          res_path,
          "/results_30seeds_testing_2001-2023_",
          season, "_", state, "_month_", period, "_", n_bins, "bins.csv"
        ),
        row.names = FALSE
      )

      results_summary_df <- results_df %>%
        group_by(Target, experiment) %>%
        summarize(
          Avg_Accuracy = mean(Accuracy, na.rm = TRUE),
          Avg_Macro_Precision = mean(Macro_Precision, na.rm = TRUE),
          Avg_Macro_Recall = mean(Macro_Recall, na.rm = TRUE),
          Avg_Macro_F1 = mean(Macro_F1, na.rm = TRUE),
          .groups = "drop"
        )

      write.csv(
        results_summary_df,
        paste0(
          res_path,
          "/results_summary_testing_2001-2023_",
          season, "_", state, "_month_", period, "_", n_bins, "bins.csv"
        ),
        row.names = FALSE
      )
    }
  }
}
