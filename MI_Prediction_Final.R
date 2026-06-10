#============================================================
# Predicting Myocardial Infarction from Cardiac Features
# A Multi-Modal Ablation Study (ECG vs. PPG vs. Demographics)
#
# HarvardX PH125.9x Data Science Capstone - Choose Your Own Project
# Author: Jim Titus
#============================================================
# This script is the standalone companion to the .Rmd report. It runs the full
# analysis end to end and prints every result that appears in the report, namely
# the data integrity checks, the train/test split summary, and the ablation table.
#
# Origin of the project. It started from the HEART score, a clinical tool that
# stratifies a chest-pain patient's risk from five inputs (History, ECG, Age, Risk
# factors, Troponin). HEART groups disparate predictors into clinical categories,
# and that idea shaped both the dataset choice and the bucketing used here. A first
# model built on the HEART-aligned features predicted MI almost perfectly, about
# 99.9% accuracy, which prompted the real question. How much of that accuracy
# survives once the diagnostic ECG channel is removed? That question is what this
# ablation answers. The ECG morphology features encode the textbook diagnostic
# markers of MI, and the synthetic generator encodes them more cleanly than real
# physiology would, so detecting MI from a clean ECG is trivial. The useful question
# is how much the non-diagnostic, wearable-grade PPG features plus demographics can
# recover on their own.
#============================================================


#============================================================
# Setup. Auto-install missing packages, then load.
#============================================================
if (!require(caret, quietly = TRUE))        { install.packages("caret"); library(caret) }
if (!require(randomForest, quietly = TRUE)) { install.packages("randomForest"); library(randomForest) }
if (!require(ggplot2, quietly = TRUE))      { install.packages("ggplot2"); library(ggplot2) }


#============================================================
# Load data from GitHub (auto-download, relative path, no login)
#============================================================
data_url   <- paste0("https://raw.githubusercontent.com/leftbrain847/",
                     "Myocardial_Infarction_Prediction/main/Data/combined_cardiac_dataset.csv")
local_copy <- "combined_cardiac_dataset.csv"
if (!file.exists(local_copy)) download.file(data_url, local_copy, mode = "wb")
cardiac <- read.csv(local_copy, stringsAsFactors = FALSE)

cat("Raw data dimensions:", nrow(cardiac), "rows x", ncol(cardiac), "cols\n")


#============================================================
# Cleanup and target preparation
#============================================================
# Drop patient_id (a pure identifier) and mi_subtype (a finer-grained version of
# the target, STEMI/NSTEMI/Old_MI, that would leak the outcome if kept).
df <- cardiac[, !(names(cardiac) %in% c("patient_id", "mi_subtype"))]

# Target to factor with "Yes" as the positive class so caret reports sensitivity
# and specificity against the clinically meaningful class.
df$mi_label <- factor(ifelse(df$mi_label == 1, "Yes", "No"), levels = c("No", "Yes"))

# Encode character categoricals (e.g. sex, smoking_status) as factors.
chr_cols <- names(df)[sapply(df, is.character)]
df[chr_cols] <- lapply(df[chr_cols], factor)

# No missing values in this synthetic set, so no imputation is needed.
cat("Total missing values:", sum(is.na(df)), "\n")
cat("MI prevalence:", round(mean(df$mi_label == "Yes"), 4), "\n")


#============================================================
# Feature bucketing (the core design choice)
#============================================================
# Every predictor is assigned to exactly one bucket. Models are then defined by
# which buckets are "in" (see build_feature_set() and the ablation below).
#
# Two bucketing decisions make the central comparison harder rather than easier,
# and therefore more honest.
#  - HRV metrics go in the ECG bucket because they are computed from the ECG R-R
#    intervals.
#  - pulse_rate_bpm and peak_to_peak_interval_ms are measured by the PPG sensor but
#    carry the same heart-rate information as the ECG and are collinear with it, so
#    they are assigned to ECG. This denies the PPG-only model heart-rate access, so
#    any performance it keeps comes from genuinely PPG-specific information rather
#    than a back-door copy of heart rate.
#  - signal_quality holds measurement metadata that describes the recording rather
#    than the patient. Letting it predict MI would be an artifact, so it is excluded
#    from every model.
buckets <- list(

  demographics = c("age", "sex", "bmi", "smoking_status", "diabetes",
                   "hypertension", "total_cholesterol_mg_dl", "family_history_cad"),

  ecg = c("heart_rate_bpm", "rr_interval_ms", "pr_interval_ms", "qrs_duration_ms",
          "qt_interval_ms", "qtc_bazett_ms", "st_elevation_mv", "st_depression_mv",
          "st_slope", "t_wave_amplitude_mv", "t_wave_inversion", "pathological_q_wave",
          "p_wave_duration_ms", "p_wave_amplitude_mv",
          # HRV is ECG-derived:
          "hrv_sdnn_ms", "hrv_rmssd_ms", "hrv_lf_power_ms2", "hrv_hf_power_ms2",
          "hrv_lf_hf_ratio"),

  ppg = c("spo2_percent", "pulse_transit_time_ms", "augmentation_index_pct",
          "stiffness_index_m_s", "reflection_index_pct", "systolic_peak_amplitude",
          "diastolic_peak_amplitude", "pulse_wave_velocity_m_s", "perfusion_index_pct",
          "systolic_upstroke_time_ms", "diastolic_time_ms", "crest_time_ratio"),

  signal_quality = c("signal_quality_score", "noise_level_db", "recording_lead",
                     "ppg_signal_quality", "motion_artifact_score", "measurement_site")
)

# pulse_rate_bpm and peak_to_peak_interval_ms are PPG-measured but heart-rate-
# equivalent and collinear with ECG heart rate, so they go to ECG (the deliberate
# handicap on the PPG-only model explained above).
buckets$ecg <- c(buckets$ecg, "pulse_rate_bpm", "peak_to_peak_interval_ms")

# Integrity check. Every predictor must land in exactly one bucket, none missing
# and none duplicated. If this fails the analysis stops rather than silently
# dropping or double-counting a feature.
all_bucketed <- unlist(buckets, use.names = FALSE)
all_preds    <- setdiff(names(df), "mi_label")
stopifnot(setequal(all_bucketed, all_preds))
stopifnot(!any(duplicated(all_bucketed)))
cat("Bucket integrity OK. Feature counts per bucket:\n")
print(sapply(buckets, length))


#============================================================
# Exploratory data analysis (figures used in the report)
#============================================================
# Saved to PNG when run as a script. The .Rmd renders the same plots inline.

# 1. Class balance. Establishes that accuracy must be read against the 0.815 NIR.
ggsave("fig_balance.png",
  ggplot(df, aes(mi_label, fill = mi_label)) +
    geom_bar(colour = "black") +
    scale_fill_manual(values = c("No" = "lightsteelblue", "Yes" = "indianred")) +
    labs(title = "Class balance of MI label", x = "MI", y = "Patients") +
    theme_classic() + theme(legend.position = "none"),
  width = 5, height = 3.2, dpi = 120)

# 2. ECG separation (ST elevation). Why any ECG-bearing model reaches about 99.9%.
ggsave("fig_ecg_sep.png",
  ggplot(df, aes(st_elevation_mv, fill = mi_label)) +
    geom_density(alpha = 0.5) +
    scale_fill_manual(values = c("No" = "lightsteelblue", "Yes" = "indianred")) +
    labs(title = "ST elevation by MI label (ECG feature)",
         x = "ST elevation (mV)", y = "Density", fill = "MI") +
    theme_classic(),
  width = 6, height = 3.2, dpi = 120)

# 3. PPG separation (pulse-wave velocity). Real but weaker than ECG.
ggsave("fig_ppg_sep.png",
  ggplot(df, aes(pulse_wave_velocity_m_s, fill = mi_label)) +
    geom_density(alpha = 0.5) +
    scale_fill_manual(values = c("No" = "lightsteelblue", "Yes" = "indianred")) +
    labs(title = "Pulse wave velocity by MI label (PPG feature)",
         x = "PWV (m/s)", y = "Density", fill = "MI") +
    theme_classic(),
  width = 6, height = 3.2, dpi = 120)

# 4. Demographic weakness (age). Heavy overlap, foreshadows a weak demo-only model.
ggsave("fig_demo_age.png",
  ggplot(df, aes(age, fill = mi_label)) +
    geom_density(alpha = 0.5) +
    scale_fill_manual(values = c("No" = "lightsteelblue", "Yes" = "indianred")) +
    labs(title = "Age by MI label (demographic feature)",
         x = "Age", y = "Density", fill = "MI") +
    theme_classic(),
  width = 6, height = 3.2, dpi = 120)
cat("EDA figures written (fig_balance / fig_ecg_sep / fig_ppg_sep / fig_demo_age).\n")


#============================================================
# Train / test split (anti-leak boundary, stratified on outcome)
#============================================================
# 80/20 split. With 5,000 rows, a 20% test set holds about 1,000 patients and
# roughly 185 MI cases, enough for stable minority-class metrics like sensitivity,
# while 80% training gives the random forest ample data. A 50/50 split would starve
# the models, and a 90/10 split would leave too few MI cases for stable sensitivity.
# In createDataPartition the argument p sets the proportion placed in the returned
# index, and that index becomes the test set here, so p = 0.2 yields a 20% test set.
# sample.kind = "Rounding" keeps the split reproducible across R versions.
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(df$mi_label, times = 1, p = 0.2, list = FALSE)
train <- df[-test_index, ]
test  <- df[test_index, ]

# Align test factor levels to those learned on train so prediction never meets an
# unseen category. The stopifnot confirms this introduces no NA.
for (c in chr_cols) test[[c]] <- factor(test[[c]], levels = levels(train[[c]]))
stopifnot(!any(is.na(test)))

cat("train rows:", nrow(train), " test rows:", nrow(test), "\n")
cat("train MI prevalence:", round(mean(train$mi_label == "Yes"), 4),
    " test MI prevalence:", round(mean(test$mi_label == "Yes"), 4), "\n")


#============================================================
# Helpers
#============================================================
# Union of the columns in the requested buckets.
build_feature_set <- function(bucket_names) {
  unlist(buckets[bucket_names], use.names = FALSE)
}

# Five-fold CV with class probabilities for AUC scoring.
control <- trainControl(method = "cv", number = 5,
                        classProbs = TRUE, summaryFunction = twoClassSummary)

# AUC via the Mann-Whitney U identity (no extra package dependency).
auc_score <- function(prob, truth) {
  r  <- rank(prob)
  n1 <- sum(truth == "Yes")
  n0 <- sum(truth == "No")
  (sum(r[truth == "Yes"]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# Train logistic + random forest on a feature set, evaluate on the test set, and
# return one tidy row per model. Both models share identical seeds and data for
# every set, so any performance difference is due to the model or the features, not
# the random draw. Metrics are imbalance-aware. Accuracy is read against the NIR,
# and sensitivity is the clinically critical metric (catching true MIs).
run_models <- function(feature_cols, set_label) {
  tr <- train[, c(feature_cols, "mi_label")]
  te <- test[,  c(feature_cols, "mi_label")]

  # Logistic regression (interpretable baseline). On ECG-bearing sets the classes
  # are nearly separable, so glm emits a "fitted probabilities 0 or 1" warning,
  # the expected symptom of perfect separation rather than an error.
  set.seed(1)
  glm_fit  <- train(mi_label ~ ., data = tr, method = "glm",
                    family = "binomial", trControl = control, metric = "ROC")
  glm_pred <- predict(glm_fit, te)
  glm_prob <- predict(glm_fit, te, type = "prob")[, "Yes"]
  glm_cm   <- confusionMatrix(glm_pred, te$mi_label, positive = "Yes")

  # Random forest (the more advanced model, stable under the separation above).
  set.seed(1)
  rf_fit   <- randomForest(mi_label ~ ., data = tr)
  rf_pred  <- predict(rf_fit, te)
  rf_prob  <- predict(rf_fit, te, type = "prob")[, "Yes"]
  rf_cm    <- confusionMatrix(rf_pred, te$mi_label, positive = "Yes")

  grab <- function(cm, prob, model) {
    data.frame(
      feature_set = set_label,
      model       = model,
      n_features  = length(feature_cols),
      accuracy    = unname(cm$overall["Accuracy"]),
      sensitivity = unname(cm$byClass["Sensitivity"]),
      specificity = unname(cm$byClass["Specificity"]),
      balanced    = unname(cm$byClass["Balanced Accuracy"]),
      kappa       = unname(cm$overall["Kappa"]),
      AUC         = auc_score(prob, te$mi_label),
      row.names   = NULL
    )
  }
  rbind(grab(glm_cm, glm_prob, "logistic"),
        grab(rf_cm,  rf_prob,  "randomForest"))
}


#============================================================
# The ablation. Define model sets by bucket composition, run, compare.
#============================================================
# No-information rate on the test set. This is the accuracy of always predicting
# the majority class, and every model's accuracy should be read against it.
NIR <- max(prop.table(table(test$mi_label)))
cat("\nTest-set no-information rate (always predict 'No'):", round(NIR, 4), "\n\n")

# Each entry lists which buckets are "in". signal_quality is excluded everywhere
# (it would be an artifact). Easy to add or remove sets here.
model_sets <- list(
  "Full (ECG+PPG+Demo)" = c("ecg", "ppg", "demographics"),
  "ECG only"            = c("ecg"),
  "PPG + Demographics"  = c("ppg", "demographics"),
  "Demographics only"   = c("demographics")
)

results <- do.call(rbind, lapply(names(model_sets), function(label) {
  run_models(build_feature_set(model_sets[[label]]), label)
}))

cat("===== ABLATION RESULTS (held-out test set) =====\n")
print(format(results, digits = 4), row.names = FALSE)


#============================================================
# Summary of findings
#============================================================
# - Full and ECG-only are near-perfect (accuracy and AUC at their ceilings). ECG
#   carries nearly all the recoverable information, so detecting MI from a clean
#   (synthetic) ECG is trivial.
# - PPG + Demographics is the substantive result. With ECG and heart rate withheld,
#   it still reaches about 97 to 98% accuracy, about 91% sensitivity, and AUC near
#   0.99. Wearable-grade measurement recovers most of the discriminative power of a
#   full ECG.
# - Demographics only reaches about 81% accuracy, which only matches the 0.815 NIR.
#   It catches about 5 to 10% of MIs (kappa near 0, AUC about 0.70 to 0.74). This is
#   the accuracy paradox, and the clearest reason to report sensitivity and AUC
#   rather than accuracy alone.
#
# Limitations. The data is synthetic (separation is cleaner than real physiology),
# there is no troponin biomarker, and the buckets are treated as cleanly separable
# modalities. See the report's Conclusion for the full discussion and future work.
#============================================================
