import type { HealthResponse, LegacyAnalysisResponse, LegacyDatasetResponse, LegacyTrainingResponse, ModelInfo, PredictionResponse } from "@/types/api";

const apiUrl = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${apiUrl}${path}`, { ...options, cache: "no-store" });
  } catch (error) {
    console.error("FastAPI connection error", error);
    throw new Error("CONNECTION_ERROR");
  }

  if (!response.ok) {
    let detail = "";
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? "";
    } catch {
      detail = response.statusText;
    }
    console.error("FastAPI error", response.status, detail);
    throw new Error(`API_${response.status}`);
  }
  return response.json() as Promise<T>;
}

async function requestBlob(path: string, options?: RequestInit): Promise<Blob> {
  let response: Response;
  try {
    response = await fetch(`${apiUrl}${path}`, { ...options, cache: "no-store" });
  } catch (error) {
    console.error("FastAPI connection error", error);
    throw new Error("CONNECTION_ERROR");
  }
  if (!response.ok) throw new Error(`API_${response.status}`);
  return response.blob();
}

export const api = {
  url: apiUrl,
  health: () => request<HealthResponse>("/api/v1/health"),
  models: () => request<ModelInfo[]>("/api/v1/models"),
  legacyAnalysis: (language: string) => request<LegacyAnalysisResponse>(`/api/v1/legacy/analysis?language=${encodeURIComponent(language)}`),
  legacyTraining: () => request<LegacyTrainingResponse>("/api/v1/legacy/training"),
  legacyDataset: (language: string) => request<LegacyDatasetResponse>(`/api/v1/legacy/dataset?language=${encodeURIComponent(language)}`),
  report: (payload: PredictionResponse & { language: string }) => requestBlob("/api/v1/legacy/report", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }),
  predict: (image: File, model: string) => {
    const formData = new FormData();
    formData.append("image", image);
    return request<PredictionResponse>(`/api/v1/predict?model=${encodeURIComponent(model)}`, {
      method: "POST",
      body: formData,
    });
  },
};