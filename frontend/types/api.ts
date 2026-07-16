export interface HealthResponse {
  status: string;
  models_available: number;
  models_total: number;
}

export interface ModelInfo {
  key: string;
  name: string;
  input_width: number;
  input_height: number;
  recommended: boolean;
  available: boolean;
}

export interface PredictionResponse {
  model: string;
  model_key: string;
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface LegacyBinomialTest {
  accuracy: number;
  correct_predictions: number;
  total_samples: number;
  expected_accuracy: number;
  p_value: number;
  significant: boolean;
  confidence_interval_95: {
    lower: number | null;
    upper: number | null;
    method: string;
  };
  interpretation: string;
}

export interface LegacyModelAnalysis {
  name: string;
  model_key: string;
  classes: string[];
  confusion_matrix: number[][];
  roc: {
    fpr: number[];
    tpr: number[];
    auc: number;
  };
  mcc: number;
  binomial_accuracy_test: LegacyBinomialTest;
  architecture: string[];
  validation_accuracy: number;
  validation_loss: number;
  training_time_hours: number;
}

export interface LegacyComparisonRow {
  name: string;
  model_key: string;
  validation_accuracy: number;
  validation_loss: number;
  training_time_hours: number;
}

export interface LegacyAnalysisResponse {
  source: string;
  classes: string[];
  models: LegacyModelAnalysis[];
  model_comparison: LegacyComparisonRow[];
  mcnemar_test: {
    table: (string | number)[][];
    chi2: number;
    p_value: number | null;
    method: string;
    models: string[];
  };
}

export interface LegacyTrainingResponse {
  source: string;
  plot_available: boolean;
  plot_endpoint: string | null;
  comparison: LegacyComparisonRow[];
}

export interface LegacyDatasetResponse {
  source: string;
  dataset: string;
  classes: number;
  images: number;
  resolution: string;
  links: Record<string, string>;
  summary: string;
}

export interface ApiError {
  status: number;
  message: string;
}