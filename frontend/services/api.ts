import type { HealthResponse, ModelInfo, PredictionResponse } from "@/types/api";

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

function imageForm(image: File) {
  const formData = new FormData();
  formData.append("image", image);
  return formData;
}

export const api = {
  health: () => request<HealthResponse>("/api/v1/health"),
  models: () => request<ModelInfo[]>("/api/v1/models"),
  predict: (image: File, model: string) => request<PredictionResponse>(`/api/v1/predict?model=${encodeURIComponent(model)}`, { method: "POST", body: imageForm(image) }),
  patientReport: (payload: PredictionResponse & { language: string }, image: File) => { const body = new FormData(); body.append("payload", JSON.stringify(payload)); body.append("image", image); return requestBlob("/api/v1/patient/report", { method: "POST", body }); },
  gradCam: (image: File, model: string, targetClass: string) => requestBlob(`/api/v1/scientific/explainability/grad-cam?model=${encodeURIComponent(model)}&target_class=${encodeURIComponent(targetClass)}`, { method: "POST", body: imageForm(image) }),
  gradCamReport: (image: File, model: string, targetClass: string, language: string) => requestBlob(`/api/v1/scientific/explainability/grad-cam/report?model=${encodeURIComponent(model)}&target_class=${encodeURIComponent(targetClass)}&language=${encodeURIComponent(language)}`, { method: "POST", body: imageForm(image) }),
};