"use client";

import { type DragEvent, type ReactNode, useEffect, useMemo, useState } from "react";
import { Activity, AlertCircle, Brain, CheckCircle2, ClipboardCheck, Download, Eye, FileText, ImagePlus, Languages, LoaderCircle, Microscope, Moon, ShieldCheck, Sparkles, Stethoscope, Sun, UploadCloud } from "lucide-react";
import { api } from "@/services/api";
import type { ModelInfo, PredictionResponse } from "@/types/api";
import es from "@/messages/es.json";
import en from "@/messages/en.json";

type Language = "es" | "en";
type Connection = "checking" | "connected" | "disconnected";

const MAX_FILE_SIZE = 15 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/webp"];

function messageFor(error: unknown, t: typeof es) {
  const code = error instanceof Error ? error.message : "";
  if (code === "CONNECTION_ERROR") return t.connectionError;
  if (code === "API_400") return t.api400;
  if (code === "API_413") return t.api413;
  if (code === "API_415") return t.api415;
  if (code === "API_422") return t.api422;
  if (code === "API_503") return t.api503;
  return t.apiError;
}

function probabilityPercent(value: number) {
  return `${value.toFixed(2)}%`;
}

function fileSizeLabel(bytes: number) {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function confidenceProfile(confidence: number, t: typeof es) {
  if (confidence >= 70) return { label: t.confidenceHigh, note: t.confidenceHighNote, tone: "high" as const };
  if (confidence >= 40) return { label: t.confidenceModerate, note: t.confidenceModerateNote, tone: "moderate" as const };
  return { label: t.confidenceLow, note: t.confidenceLowNote, tone: "low" as const };
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export default function DiagnosisDashboard() {
  const [language, setLanguage] = useState<Language>("es");
  const [dark, setDark] = useState(false);
  const [connection, setConnection] = useState<Connection>("checking");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelKey, setModelKey] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [gradCamUrl, setGradCamUrl] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const [gradCamLoading, setGradCamLoading] = useState(false);
  const [gradCamReportLoading, setGradCamReportLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const t = language === "es" ? es : en;

  useEffect(() => {
    const saved = window.localStorage.getItem("theme") === "dark";
    setDark(saved);
    document.documentElement.classList.toggle("dark", saved);
  }, []);

  useEffect(() => {
    let active = true;
    Promise.all([api.health(), api.models()])
      .then(([health, availableModels]) => {
        if (!active) return;
        setConnection(health.status === "ok" ? "connected" : "disconnected");
        const usable = availableModels.filter((item) => item.available);
        setModels(usable);
        setModelKey((current) => current || usable.find((item) => item.recommended)?.key || usable[0]?.key || "");
        if (!usable.length) setError(t.noModels);
      })
      .catch((requestError) => {
        if (!active) return;
        setConnection("disconnected");
        setError(messageFor(requestError, t));
      });
    return () => { active = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  useEffect(() => () => {
    if (gradCamUrl) URL.revokeObjectURL(gradCamUrl);
  }, [gradCamUrl]);

  const selectedModel = useMemo(() => models.find((item) => item.key === modelKey), [models, modelKey]);
  const orderedProbabilities = useMemo(() => Object.entries(result?.probabilities ?? {}).sort((a, b) => b[1] - a[1]), [result]);
  const confidence = result ? confidenceProfile(result.confidence, t) : null;
  const statusText = result ? t.analysisComplete : file ? t.readyForAnalysis : t.readyForImage;

  const toggleTheme = () => {
    const next = !dark;
    setDark(next);
    window.localStorage.setItem("theme", next ? "dark" : "light");
    document.documentElement.classList.toggle("dark", next);
  };

  const resetExplainability = () => {
    if (gradCamUrl) URL.revokeObjectURL(gradCamUrl);
    setGradCamUrl("");
  };

  const chooseFile = (selected: File | undefined) => {
    setError("");
    setResult(null);
    resetExplainability();
    if (!selected) return;
    if (!ACCEPTED_TYPES.includes(selected.type)) return setError(t.invalidFile);
    if (selected.size > MAX_FILE_SIZE) return setError(t.fileTooLarge);
    if (preview) URL.revokeObjectURL(preview);
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };

  const handleDragOver = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    chooseFile(event.dataTransfer.files?.[0]);
  };

  const analyze = async () => {
    if (!file) return setError(t.noImage);
    if (!modelKey) return setError(t.noModels);
    setError("");
    setLoading(true);
    setResult(null);
    resetExplainability();
    try {
      setResult(await api.predict(file, modelKey));
    } catch (requestError) {
      setError(messageFor(requestError, t));
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async () => {
    if (!file || !result) return;
    setReportLoading(true);
    setError("");
    try {
      downloadBlob(await api.patientReport({ ...result, language }, file), `diagnosis_result_${result.model_key}.pdf`);
    } catch (requestError) {
      setError(messageFor(requestError, t));
    } finally {
      setReportLoading(false);
    }
  };

  const showGradCam = async () => {
    if (!file || !result) return;
    setGradCamLoading(true);
    setError("");
    try {
      const blob = await api.gradCam(file, result.model_key, result.predicted_class);
      resetExplainability();
      setGradCamUrl(URL.createObjectURL(blob));
    } catch (requestError) {
      const code = requestError instanceof Error ? requestError.message : "";
      setError(code === "API_422" ? t.gradCamUnavailable : messageFor(requestError, t));
    } finally {
      setGradCamLoading(false);
    }
  };

  const downloadGradCamReport = async () => {
    if (!file || !result) return;
    setGradCamReportLoading(true);
    setError("");
    try {
      downloadBlob(await api.gradCamReport(file, result.model_key, result.predicted_class, language), `grad_cam_${result.model_key}.pdf`);
    } catch (requestError) {
      setError(messageFor(requestError, t));
    } finally {
      setGradCamReportLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-100 text-slate-950 dark:bg-slate-950 dark:text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-4 sm:px-6 lg:px-8">
        <header className="rounded-lg border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <div className="flex flex-col gap-5 p-5 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex min-w-0 items-center gap-4">
              <span className="grid h-12 w-12 shrink-0 place-items-center rounded-md bg-teal-700 text-white"><Stethoscope size={24} /></span>
              <div className="min-w-0">
                <p className="text-xs font-semibold uppercase text-teal-700 dark:text-teal-300">{t.clinicalConsole}</p>
                <h1 className="mt-1 text-2xl font-semibold tracking-normal sm:text-3xl">{t.appName}</h1>
                <p className="mt-1 max-w-3xl text-sm text-slate-600 dark:text-slate-300">{t.clinicalConsoleBody}</p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <ConnectionBadge connection={connection} t={t} />
              <span className="inline-flex h-10 items-center rounded-md border border-slate-200 px-3 text-sm font-medium dark:border-slate-700">{t.caseStatus}: {statusText}</span>
              <Languages size={18} className="text-slate-500" />
              <select aria-label={t.language} value={language} onChange={(event) => setLanguage(event.target.value as Language)} className="h-10 rounded-md border border-slate-300 bg-white px-3 text-sm dark:border-slate-700 dark:bg-slate-950">
                <option value="es">Espanol</option><option value="en">English</option>
              </select>
              <button type="button" onClick={toggleTheme} title={dark ? t.light : t.dark} className="grid h-10 w-10 place-items-center rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200 dark:hover:bg-slate-800">
                {dark ? <Sun size={18} /> : <Moon size={18} />}
              </button>
            </div>
          </div>
        </header>

        {error && <div role="alert" className="mt-4 flex gap-3 rounded-md border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900 dark:border-rose-900 dark:bg-rose-950 dark:text-rose-100"><AlertCircle className="shrink-0" size={19} /><span>{error}</span></div>}

        <section aria-label={t.workflowTitle} className="mt-4 grid gap-3 lg:grid-cols-3">
          <WorkflowStep icon={<ClipboardCheck size={18} />} title={t.imageQuality} body={t.imageQualityBody} />
          <WorkflowStep icon={<Brain size={18} />} title={t.modelSelection} body={t.modelSelectionBody} />
          <WorkflowStep icon={<FileText size={18} />} title={t.reviewAndReport} body={t.reviewAndReportBody} />
        </section>

        <section className="mt-4 grid flex-1 gap-4 xl:grid-cols-[minmax(320px,410px)_minmax(0,1fr)]">
          <aside className="space-y-4">
            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
              <div className="mb-4 flex items-start justify-between gap-3"><div><h2 className="text-lg font-semibold">{t.uploadPanelTitle}</h2><p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{t.uploadPanelBody}</p></div><Microscope className="shrink-0 text-teal-700 dark:text-teal-300" size={24} /></div>
              <label className={`flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-5 text-center transition-colors ${isDragging ? "border-teal-500 bg-teal-50 text-teal-950 dark:bg-teal-950/60 dark:text-teal-100" : "border-slate-300 bg-slate-50 hover:border-teal-600 hover:bg-teal-50 dark:border-slate-700 dark:bg-slate-950 dark:hover:border-teal-400 dark:hover:bg-teal-950/40"}`} onDragEnter={handleDragOver} onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
                <UploadCloud size={34} className={isDragging ? "text-teal-700 dark:text-teal-200" : "text-slate-500 dark:text-slate-400"} />
                <span className="mt-3 font-semibold">{isDragging ? t.dropImage : file ? t.changeImage : t.chooseImage}</span>
                <span className="mt-1 max-w-xs text-sm text-slate-600 dark:text-slate-300">{t.uploadHint}</span>
                <input className="sr-only" type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => chooseFile(event.target.files?.[0])} />
              </label>
              {preview && <div className="mt-4 overflow-hidden rounded-lg border border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-950"><div className="flex items-center justify-between gap-3 border-b border-slate-200 px-3 py-2 text-sm dark:border-slate-800"><span className="font-medium">{t.selectedImage}</span><span className="truncate text-slate-500">{file?.name}</span></div><img src={preview} alt={t.preview} className="max-h-72 w-full object-contain" /><div className="flex justify-between border-t border-slate-200 px-3 py-2 text-xs text-slate-500 dark:border-slate-800"><span>{t.fileReady}</span><span>{file ? fileSizeLabel(file.size) : ""}</span></div></div>}
            </section>

            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
              <h2 className="text-lg font-semibold">{t.modelPanelTitle}</h2>
              <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{t.modelPanelBody}</p>
              <label className="mb-2 mt-4 block text-sm font-medium" htmlFor="model">{t.models}</label>
              <select id="model" value={modelKey} onChange={(event) => setModelKey(event.target.value)} disabled={!models.length || loading} className="h-11 w-full rounded-md border border-slate-300 bg-white px-3 text-sm dark:border-slate-700 dark:bg-slate-950">
                {models.map((item) => <option key={item.key} value={item.key}>{item.name}{item.recommended ? ` - ${t.recommended}` : ""}</option>)}
              </select>
              <div className="mt-3 rounded-md bg-slate-50 p-3 text-sm text-slate-600 dark:bg-slate-950 dark:text-slate-300">
                <div className="font-medium text-slate-900 dark:text-slate-100">{selectedModel?.name ?? t.noModels}</div>
                <div>{selectedModel ? `${selectedModel.input_width} x ${selectedModel.input_height}` : t.noModels}</div>
              </div>
              <button type="button" onClick={analyze} disabled={!file || !modelKey || loading} className="mt-4 flex h-12 w-full items-center justify-center gap-2 rounded-md bg-teal-700 px-4 font-semibold text-white enabled:hover:bg-teal-800 disabled:cursor-not-allowed disabled:bg-slate-400">{loading ? <LoaderCircle className="animate-spin" size={18} /> : <ImagePlus size={18} />}{loading ? t.analyzing : result ? t.runNewAnalysis : t.analyze}</button>
            </section>
          </aside>

          <ResultPanel t={t} result={result} probabilities={orderedProbabilities} confidence={confidence} reportLoading={reportLoading} gradCamLoading={gradCamLoading} gradCamReportLoading={gradCamReportLoading} gradCamUrl={gradCamUrl} downloadReport={downloadReport} showGradCam={showGradCam} downloadGradCamReport={downloadGradCamReport} />
        </section>

        <section className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-950 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-100">
          <div className="flex items-start gap-3"><ShieldCheck className="mt-0.5 shrink-0" size={18} /><div><h2 className="font-semibold">{t.medicalDisclaimerTitle}</h2><p className="mt-1">{t.medicalDisclaimerBody}</p></div></div>
        </section>
      </div>
    </main>
  );
}

function WorkflowStep({ icon, title, body }: { icon: ReactNode; title: string; body: string }) {
  return <div className="flex min-h-28 gap-3 rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900"><span className="grid h-10 w-10 shrink-0 place-items-center rounded-md bg-teal-50 text-teal-700 dark:bg-teal-950 dark:text-teal-200">{icon}</span><div><h2 className="text-sm font-semibold">{title}</h2><p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{body}</p></div></div>;
}

function ConnectionBadge({ connection, t }: { connection: Connection; t: typeof es }) {
  const connected = connection === "connected";
  const checking = connection === "checking";
  return <span className={`inline-flex h-10 items-center gap-2 rounded-md px-3 text-sm font-medium ${connected ? "bg-emerald-50 text-emerald-800 ring-1 ring-emerald-200 dark:bg-emerald-950 dark:text-emerald-200 dark:ring-emerald-900" : "bg-rose-50 text-rose-800 ring-1 ring-rose-200 dark:bg-rose-950 dark:text-rose-200 dark:ring-rose-900"}`}>{checking ? <LoaderCircle className="animate-spin" size={16} /> : connected ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}<span>{checking ? t.checking : connected ? t.connected : t.disconnected}</span></span>;
}

function ResultPanel({ t, result, probabilities, confidence, reportLoading, gradCamLoading, gradCamReportLoading, gradCamUrl, downloadReport, showGradCam, downloadGradCamReport }: { t: typeof es; result: PredictionResponse | null; probabilities: [string, number][]; confidence: ReturnType<typeof confidenceProfile> | null; reportLoading: boolean; gradCamLoading: boolean; gradCamReportLoading: boolean; gradCamUrl: string; downloadReport: () => void; showGradCam: () => void; downloadGradCamReport: () => void }) {
  if (!result) {
    return <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900"><div className="flex min-h-[620px] flex-col items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50 p-6 text-center dark:border-slate-700 dark:bg-slate-950"><Microscope size={44} className="text-slate-400" /><h2 className="mt-5 text-lg font-semibold">{t.result}</h2><p className="mt-2 max-w-md text-sm text-slate-600 dark:text-slate-300">{t.emptyResult}</p></div></section>;
  }
  const topAlternatives = probabilities.slice(1, 4);
  const toneClass = confidence?.tone === "high" ? "border-emerald-200 bg-emerald-50 text-emerald-950 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-100" : confidence?.tone === "moderate" ? "border-amber-200 bg-amber-50 text-amber-950 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-100" : "border-rose-200 bg-rose-50 text-rose-950 dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-100";
  return <section className="space-y-4">
    <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between"><div><p className="text-xs font-semibold uppercase text-teal-700 dark:text-teal-300">{t.primaryFinding}</p><h2 className="mt-2 text-3xl font-semibold tracking-normal">{t.classNames[result.predicted_class as keyof typeof t.classNames] ?? result.predicted_class}</h2><p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{result.predicted_class} | {result.model}</p></div><div className={`rounded-lg border p-4 ${toneClass}`}><p className="text-xs font-semibold uppercase">{t.confidenceLevel}</p><p className="mt-1 text-2xl font-semibold">{probabilityPercent(result.confidence)}</p><p className="mt-1 text-sm font-medium">{confidence?.label}</p></div></div>
      <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-300"><p className="font-semibold text-slate-950 dark:text-slate-100">{t.clinicalReading}</p><p className="mt-1">{confidence?.note ?? t.clinicalReadingBody}</p></div>
    </div>

    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_300px]">
      <section className="rounded-lg border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900"><div className="border-b border-slate-200 px-4 py-3 dark:border-slate-800"><h3 className="text-sm font-semibold">{t.probabilityDistribution}</h3></div><div className="space-y-3 p-4">{probabilities.map(([className, probability], index) => <div key={className}><div className="mb-1 flex justify-between gap-3 text-sm"><span className="min-w-0 truncate"><span className="font-medium">{index + 1}. {t.classNames[className as keyof typeof t.classNames] ?? className}</span> <span className="text-slate-500">{className}</span></span><span className="shrink-0 font-semibold">{probabilityPercent(probability)}</span></div><div className="h-2.5 rounded-full bg-slate-200 dark:bg-slate-800"><div className="h-2.5 rounded-full bg-teal-700" style={{ width: `${Math.max(0, Math.min(probability, 100))}%` }} /></div></div>)}</div></section>
      <aside className="space-y-4"><section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900"><h3 className="text-sm font-semibold">{t.topAlternatives}</h3><div className="mt-3 space-y-2">{topAlternatives.map(([className, probability]) => <div key={className} className="flex justify-between gap-3 rounded-md bg-slate-50 px-3 py-2 text-sm dark:bg-slate-950"><span className="truncate">{className}</span><span className="font-semibold">{probabilityPercent(probability)}</span></div>)}</div></section><section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900"><h3 className="text-sm font-semibold">{t.technicalTrace}</h3><dl className="mt-3 space-y-2 text-sm"><div className="flex justify-between gap-3"><dt className="text-slate-500">{t.model}</dt><dd className="text-right font-medium">{result.model_key}</dd></div><div className="flex justify-between gap-3"><dt className="text-slate-500">{t.generatedBy}</dt><dd className="text-right font-medium">FastAPI</dd></div></dl></section></aside>
    </div>

    <section className="grid gap-4 lg:grid-cols-2">
      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900"><h3 className="text-sm font-semibold">{t.reportActions}</h3><p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{t.reviewAndReportBody}</p><button type="button" onClick={downloadReport} disabled={reportLoading} className="mt-4 flex h-11 w-full items-center justify-center gap-2 rounded-md bg-teal-700 px-4 font-semibold text-white enabled:hover:bg-teal-800 disabled:cursor-not-allowed disabled:bg-slate-400">{reportLoading ? <LoaderCircle className="animate-spin" size={18} /> : <Download size={18} />}{reportLoading ? t.generatingReport : t.downloadReport}</button></div>
      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900"><h3 className="text-sm font-semibold">{t.explainabilityActions}</h3><p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{gradCamUrl ? t.explanationNotice : t.noGradCamYet}</p><button type="button" onClick={showGradCam} disabled={gradCamLoading} className="mt-4 flex h-11 w-full items-center justify-center gap-2 rounded-md border border-slate-300 px-4 font-semibold text-slate-800 enabled:hover:bg-slate-50 disabled:cursor-not-allowed disabled:text-slate-400 dark:border-slate-700 dark:text-slate-100 dark:enabled:hover:bg-slate-800">{gradCamLoading ? <LoaderCircle className="animate-spin" size={18} /> : <Eye size={18} />}{gradCamLoading ? t.generatingGradCam : t.showGradCam}</button></div>
    </section>

    {gradCamUrl && <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900"><div className="flex flex-wrap items-center justify-between gap-3"><div><h3 className="text-sm font-semibold">{t.explainability}</h3><p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{t.explanationNotice}</p></div><button type="button" onClick={downloadGradCamReport} disabled={gradCamReportLoading} className="flex h-10 items-center justify-center gap-2 rounded-md border border-slate-300 px-3 text-sm font-semibold enabled:hover:bg-slate-50 disabled:cursor-not-allowed disabled:text-slate-400 dark:border-slate-700 dark:enabled:hover:bg-slate-800">{gradCamReportLoading ? <LoaderCircle className="animate-spin" size={16} /> : <Download size={16} />}{gradCamReportLoading ? t.generatingExplanationReport : t.downloadExplanationReport}</button></div><img src={gradCamUrl} alt={t.explainability} className="mt-4 max-h-[560px] w-full rounded-md border border-slate-200 object-contain dark:border-slate-800" /></section>}
  </section>;
}
