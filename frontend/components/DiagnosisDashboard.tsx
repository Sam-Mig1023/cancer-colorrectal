"use client";

import { type DragEvent, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Database,
  Download,
  FileText,
  ImagePlus,
  Languages,
  LineChart,
  LoaderCircle,
  Microscope,
  Moon,
  ShieldCheck,
  Sun,
  UploadCloud,
} from "lucide-react";
import { api } from "@/services/api";
import type {
  LegacyAnalysisResponse,
  LegacyDatasetResponse,
  LegacyModelAnalysis,
  LegacyTrainingResponse,
  ModelInfo,
  PredictionResponse,
} from "@/types/api";
import es from "@/messages/es.json";
import en from "@/messages/en.json";

type Language = "es" | "en";
type Connection = "checking" | "connected" | "disconnected";
type TabKey = "diagnosis" | "analysis" | "training" | "dataset";

const MAX_FILE_SIZE = 15 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/webp"];

function messageFor(error: unknown, t: typeof es) {
  const code = error instanceof Error ? error.message : "";
  if (code === "CONNECTION_ERROR") return t.connectionError;
  if (code === "API_400") return t.api400;
  if (code === "API_413") return t.api413;
  if (code === "API_415") return t.api415;
  if (code === "API_503") return t.api503;
  return t.apiError;
}

function percent(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function probabilityPercent(value: number) {
  return `${value.toFixed(2)}%`;
}

export default function DiagnosisDashboard() {
  const [language, setLanguage] = useState<Language>("es");
  const [dark, setDark] = useState(false);
  const [activeTab, setActiveTab] = useState<TabKey>("diagnosis");
  const [connection, setConnection] = useState<Connection>("checking");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelKey, setModelKey] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [legacy, setLegacy] = useState<LegacyAnalysisResponse | null>(null);
  const [training, setTraining] = useState<LegacyTrainingResponse | null>(null);
  const [dataset, setDataset] = useState<LegacyDatasetResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
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
        if (active) {
          setConnection("disconnected");
          setError(messageFor(requestError, t));
        }
      });
    return () => { active = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let active = true;
    Promise.all([api.legacyAnalysis(language), api.legacyTraining(), api.legacyDataset(language)])
      .then(([analysis, trainingData, datasetData]) => {
        if (!active) return;
        setLegacy(analysis);
        setTraining(trainingData);
        setDataset(datasetData);
      })
      .catch((requestError) => {
        if (active) setError(messageFor(requestError, t));
      });
    return () => { active = false; };
  }, [language, t]);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  const selectedModel = useMemo(() => models.find((item) => item.key === modelKey), [models, modelKey]);
  const selectedLegacy = useMemo(() => legacy?.models.find((item) => item.model_key === modelKey), [legacy, modelKey]);
  const orderedProbabilities = useMemo(() => Object.entries(result?.probabilities ?? {}).sort((a, b) => b[1] - a[1]), [result]);
  const matrixMax = useMemo(() => Math.max(1, ...(selectedLegacy?.confusion_matrix.flat() ?? [1])), [selectedLegacy]);
  const rocPoints = useMemo(
    () => selectedLegacy?.roc.fpr.map((fpr, index) => `${fpr * 100},${100 - selectedLegacy.roc.tpr[index] * 100}`).join(" ") ?? "",
    [selectedLegacy],
  );

  const tabs = [
    { key: "diagnosis" as const, label: t.tabDiagnosis, icon: Microscope },
    { key: "analysis" as const, label: t.tabAnalysis, icon: LineChart },
    { key: "training" as const, label: t.tabTraining, icon: BarChart3 },
    { key: "dataset" as const, label: t.tabDataset, icon: Database },
  ];

  const toggleTheme = () => {
    const next = !dark;
    setDark(next);
    window.localStorage.setItem("theme", next ? "dark" : "light");
    document.documentElement.classList.toggle("dark", next);
  };

  const chooseFile = (selected: File | undefined) => {
    setError("");
    setResult(null);
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
    try {
      setResult(await api.predict(file, modelKey));
    } catch (requestError) {
      setError(messageFor(requestError, t));
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async () => {
    if (!result) return;
    setReportLoading(true);
    setError("");
    try {
      const blob = await api.report({ ...result, language });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `diagnosis_${result.model_key}.pdf`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (requestError) {
      setError(messageFor(requestError, t));
    } finally {
      setReportLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-100 text-slate-950 dark:bg-slate-950 dark:text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-4 sm:px-6 lg:px-8">
        <header className="rounded-lg border border-slate-200 bg-white px-4 py-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex min-w-0 items-center gap-3">
              <span className="grid h-11 w-11 shrink-0 place-items-center rounded-md bg-teal-700 text-white">
                <Activity size={23} />
              </span>
              <div className="min-w-0">
                <h1 className="truncate text-xl font-semibold tracking-normal sm:text-2xl">{t.appName}</h1>
                <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{t.subtitle}</p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <ConnectionBadge connection={connection} t={t} />
              <label className="sr-only" htmlFor="language">{t.language}</label>
              <Languages size={18} className="text-slate-500" />
              <select
                id="language"
                value={language}
                onChange={(event) => setLanguage(event.target.value as Language)}
                className="h-10 rounded-md border border-slate-300 bg-white px-3 text-sm dark:border-slate-700 dark:bg-slate-950"
              >
                <option value="es">Espanol</option>
                <option value="en">English</option>
              </select>
              <button
                type="button"
                onClick={toggleTheme}
                title={dark ? t.light : t.dark}
                className="grid h-10 w-10 place-items-center rounded-md border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200 dark:hover:bg-slate-800"
              >
                {dark ? <Sun size={18} /> : <Moon size={18} />}
              </button>
            </div>
          </div>
        </header>

        {error && (
          <div role="alert" className="mt-4 flex gap-3 rounded-md border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900 dark:border-rose-900 dark:bg-rose-950 dark:text-rose-100">
            <AlertCircle className="shrink-0" size={19} />
            <span>{error}</span>
          </div>
        )}

        <div className="mt-4 grid flex-1 gap-4 xl:grid-cols-[280px_minmax(0,1fr)]">
          <div className="min-w-0 xl:order-2">
            {activeTab === "diagnosis" && (
              <DiagnosisView
                t={t}
                models={models}
                modelKey={modelKey}
                selectedModel={selectedModel}
                file={file}
                preview={preview}
                result={result}
                loading={loading}
                reportLoading={reportLoading}
                isDragging={isDragging}
                orderedProbabilities={orderedProbabilities}
                setModelKey={setModelKey}
                chooseFile={chooseFile}
                handleDragOver={handleDragOver}
                handleDragLeave={handleDragLeave}
                handleDrop={handleDrop}
                analyze={analyze}
                downloadReport={downloadReport}
              />
            )}
            {activeTab === "analysis" && selectedLegacy && legacy && (
              <AnalysisView
                t={t}
                legacy={legacy}
                selectedLegacy={selectedLegacy}
                matrixMax={matrixMax}
                rocPoints={rocPoints}
              />
            )}
            {activeTab === "analysis" && (!selectedLegacy || !legacy) && <EmptyPanel title={t.legacyAnalysis} body={t.analysisUnavailable} />}
            {activeTab === "training" && <TrainingView t={t} training={training} />}
            {activeTab === "dataset" && <DatasetView t={t} dataset={dataset} />}
          </div>

          <SidebarNavigation
            t={t}
            tabs={tabs}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            selectedModel={selectedModel}
            result={result}
          />
        </div>
      </div>
    </main>
  );
}

function SidebarNavigation({
  t,
  tabs,
  activeTab,
  setActiveTab,
  selectedModel,
  result,
}: {
  t: typeof es;
  tabs: Array<{ key: TabKey; label: string; icon: typeof Microscope }>;
  activeTab: TabKey;
  setActiveTab: (tab: TabKey) => void;
  selectedModel?: ModelInfo;
  result: PredictionResponse | null;
}) {
  return (
    <aside className="order-first rounded-lg border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-800 dark:bg-slate-900 xl:order-1 xl:sticky xl:top-4 xl:self-start" aria-label={t.navigationLabel}>
      <div className="mb-3 flex items-center justify-between gap-2 px-1">
        <div>
          <p className="text-xs font-semibold uppercase text-slate-500 dark:text-slate-400">{t.navigationLabel}</p>
          <p className="mt-1 text-sm font-semibold text-slate-950 dark:text-slate-100">{t.appName}</p>
        </div>
        <Activity size={20} className="text-teal-700 dark:text-teal-300" />
      </div>

      <nav className="grid gap-2 sm:grid-cols-4 xl:grid-cols-1">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const active = activeTab === tab.key;
          return (
            <button
              key={tab.key}
              type="button"
              onClick={() => setActiveTab(tab.key)}
              className={`flex h-12 items-center justify-start gap-3 rounded-md px-3 text-sm font-semibold transition-colors ${
                active
                  ? "bg-teal-700 text-white shadow-sm"
                  : "text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
              }`}
            >
              <Icon size={18} />
              <span className="truncate">{tab.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="mt-4 hidden rounded-md border border-slate-200 bg-slate-50 p-3 text-sm dark:border-slate-800 dark:bg-slate-950 xl:block">
        <p className="text-xs font-semibold uppercase text-slate-500 dark:text-slate-400">{t.selectedModel}</p>
        <p className="mt-1 font-medium text-slate-950 dark:text-slate-100">{selectedModel?.name ?? t.noModels}</p>
        <div className="mt-3 border-t border-slate-200 pt-3 dark:border-slate-800">
          <p className="text-xs font-semibold uppercase text-slate-500 dark:text-slate-400">{t.result}</p>
          <p className="mt-1 font-medium text-slate-950 dark:text-slate-100">
            {result ? `${result.predicted_class} - ${result.confidence.toFixed(2)}%` : t.emptyResult}
          </p>
        </div>
      </div>

      <div className="mt-4 hidden items-start gap-2 rounded-md border border-teal-100 bg-teal-50 p-3 text-sm text-teal-950 dark:border-teal-900 dark:bg-teal-950/50 dark:text-teal-100 xl:flex">
        <ShieldCheck size={18} className="mt-0.5 shrink-0" />
        <span>{t.supportNotice}</span>
      </div>
    </aside>
  );
}
function ConnectionBadge({ connection, t }: { connection: Connection; t: typeof es }) {
  const connected = connection === "connected";
  const checking = connection === "checking";
  return (
    <span className={`inline-flex h-10 items-center gap-2 rounded-md px-3 text-sm font-medium ${connected ? "bg-emerald-50 text-emerald-800 ring-1 ring-emerald-200 dark:bg-emerald-950 dark:text-emerald-200 dark:ring-emerald-900" : "bg-rose-50 text-rose-800 ring-1 ring-rose-200 dark:bg-rose-950 dark:text-rose-200 dark:ring-rose-900"}`}>
      {checking ? <LoaderCircle className="animate-spin" size={16} /> : connected ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
      <span>{checking ? t.checking : connected ? t.connected : t.disconnected}</span>
    </span>
  );
}

function DiagnosisView(props: {
  t: typeof es;
  models: ModelInfo[];
  modelKey: string;
  selectedModel?: ModelInfo;
  file: File | null;
  preview: string;
  result: PredictionResponse | null;
  loading: boolean;
  reportLoading: boolean;
  isDragging: boolean;
  orderedProbabilities: [string, number][];
  setModelKey: (key: string) => void;
  chooseFile: (file: File | undefined) => void;
  handleDragOver: (event: DragEvent<HTMLLabelElement>) => void;
  handleDragLeave: (event: DragEvent<HTMLLabelElement>) => void;
  handleDrop: (event: DragEvent<HTMLLabelElement>) => void;
  analyze: () => void;
  downloadReport: () => void;
}) {
  const { t, models, modelKey, selectedModel, file, preview, result, loading, reportLoading, isDragging, orderedProbabilities, setModelKey, chooseFile, handleDragOver, handleDragLeave, handleDrop, analyze, downloadReport } = props;
  return (
    <section className="grid gap-4 xl:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold">{t.diagnosisWorkspace}</h2>
            <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{t.selectedModel}: {selectedModel?.name ?? t.noModels}</p>
          </div>
          <Microscope className="shrink-0 text-teal-700 dark:text-teal-300" size={24} />
        </div>

        <label className="mb-2 block text-sm font-medium" htmlFor="model">{t.models}</label>
        <select
          id="model"
          value={modelKey}
          onChange={(event) => setModelKey(event.target.value)}
          disabled={!models.length || loading}
          className="mb-4 h-11 w-full rounded-md border border-slate-300 bg-white px-3 text-sm dark:border-slate-700 dark:bg-slate-950"
        >
          {models.map((item) => <option key={item.key} value={item.key}>{item.name}{item.recommended ? ` - ${t.recommended}` : ""}</option>)}
        </select>

        <label
          className={`flex min-h-48 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-5 text-center transition-colors ${
            isDragging
              ? "border-teal-500 bg-teal-50 text-teal-950 dark:bg-teal-950/60 dark:text-teal-100"
              : "border-slate-300 bg-slate-50 hover:border-teal-600 hover:bg-teal-50 dark:border-slate-700 dark:bg-slate-950 dark:hover:border-teal-400 dark:hover:bg-teal-950/40"
          }`}
          onDragEnter={handleDragOver}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <UploadCloud size={32} className={isDragging ? "text-teal-700 dark:text-teal-200" : "text-slate-500 dark:text-slate-400"} />
          <span className="mt-3 font-semibold">{isDragging ? t.dropImage : file ? t.changeImage : t.chooseImage}</span>
          <span className="mt-1 max-w-xs text-sm text-slate-600 dark:text-slate-300">{t.uploadHint}</span>
          <input className="sr-only" type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => chooseFile(event.target.files?.[0])} />
        </label>

        {preview && (
          <div className="mt-4 overflow-hidden rounded-lg border border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-950">
            <div className="flex items-center justify-between border-b border-slate-200 px-3 py-2 text-sm dark:border-slate-800">
              <span className="font-medium">{t.preview}</span>
              <span className="truncate text-slate-500 dark:text-slate-400">{file?.name}</span>
            </div>
            <img src={preview} alt={t.preview} className="max-h-80 w-full object-contain" />
          </div>
        )}

        <button
          type="button"
          onClick={analyze}
          disabled={!file || !modelKey || loading}
          className="mt-4 flex h-12 w-full items-center justify-center gap-2 rounded-md bg-teal-700 px-4 font-semibold text-white enabled:hover:bg-teal-800 disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          {loading ? <LoaderCircle className="animate-spin" size={18} /> : <ImagePlus size={18} />}
          {loading ? t.analyzing : t.analyze}
        </button>
      </div>

      <ResultPanel
        t={t}
        result={result}
        orderedProbabilities={orderedProbabilities}
        reportLoading={reportLoading}
        downloadReport={downloadReport}
      />
    </section>
  );
}

function ResultPanel({ t, result, orderedProbabilities, reportLoading, downloadReport }: {
  t: typeof es;
  result: PredictionResponse | null;
  orderedProbabilities: [string, number][];
  reportLoading: boolean;
  downloadReport: () => void;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">{t.result}</h2>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{t.resultContext}</p>
        </div>
        <FileText className="shrink-0 text-teal-700 dark:text-teal-300" size={24} />
      </div>

      {!result ? (
        <div className="flex min-h-[420px] flex-col items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50 p-6 text-center dark:border-slate-700 dark:bg-slate-950">
          <Microscope size={38} className="text-slate-400" />
          <p className="mt-4 max-w-md text-sm text-slate-600 dark:text-slate-300">{t.emptyResult}</p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="grid gap-3 md:grid-cols-3">
            <MetricBlock label={t.predictedClass} value={t.classNames[result.predicted_class as keyof typeof t.classNames] ?? result.predicted_class} detail={result.predicted_class} />
            <MetricBlock label={t.confidence} value={probabilityPercent(result.confidence)} detail={t.modelOutput} tone="strong" />
            <MetricBlock label={t.model} value={result.model} detail={result.model_key} />
          </div>

          <div className="rounded-lg border border-slate-200 dark:border-slate-800">
            <div className="border-b border-slate-200 px-4 py-3 dark:border-slate-800">
              <h3 className="text-sm font-semibold">{t.probabilities}</h3>
            </div>
            <div className="space-y-3 p-4">
              {orderedProbabilities.map(([className, probability]) => (
                <div key={className}>
                  <div className="mb-1 flex justify-between gap-3 text-sm">
                    <span className="min-w-0 truncate">{t.classNames[className as keyof typeof t.classNames] ?? className} <span className="text-slate-500">{className}</span></span>
                    <span className="shrink-0 font-medium">{probabilityPercent(probability)}</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-800">
                    <div className="h-2 rounded-full bg-teal-700" style={{ width: `${Math.max(0, Math.min(probability, 100))}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <button
            type="button"
            onClick={downloadReport}
            disabled={reportLoading}
            className="flex h-11 w-full items-center justify-center gap-2 rounded-md border border-teal-700 px-4 font-semibold text-teal-800 enabled:hover:bg-teal-50 disabled:cursor-not-allowed disabled:text-slate-400 dark:text-teal-200 dark:enabled:hover:bg-teal-950"
          >
            {reportLoading ? <LoaderCircle className="animate-spin" size={18} /> : <Download size={18} />}
            {reportLoading ? t.generatingReport : t.downloadReport}
          </button>
        </div>
      )}
    </div>
  );
}

function AnalysisView({ t, legacy, selectedLegacy, matrixMax, rocPoints }: {
  t: typeof es;
  legacy: LegacyAnalysisResponse;
  selectedLegacy: LegacyModelAnalysis;
  matrixMax: number;
  rocPoints: string;
}) {
  const binomial = selectedLegacy.binomial_accuracy_test;
  const ci = binomial.confidence_interval_95;
  const ciValue = `${percent(ci.lower)} - ${percent(ci.upper)}`;
  const palette = ["#0f766e", "#2563eb", "#9333ea", "#dc2626", "#ca8a04"];

  return (
    <section className="space-y-4">
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-950 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-100">
        <strong>{t.legacyNoticeTitle}</strong> {legacy.source}. {t.legacyNoticeBody}
      </div>

      <div className="grid gap-4 lg:grid-cols-5">
        <MetricBlock label="AUC" value={selectedLegacy.roc.auc.toFixed(2)} detail={t.rocCurve} />
        <MetricBlock label="MCC" value={selectedLegacy.mcc.toFixed(4)} detail={t.mccDescription} />
        <MetricBlock label={t.observedAccuracy} value={percent(binomial.accuracy)} detail={`${binomial.correct_predictions}/${binomial.total_samples}`} />
        <MetricBlock label="p-value" value={binomial.p_value.toExponential(2)} detail={binomial.significant ? t.significant : t.notSignificant} tone={binomial.significant ? "strong" : "default"} />
        <MetricBlock label={t.confidenceInterval} value={ciValue} detail={ci.method} />
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4 text-sm shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <h2 className="text-lg font-semibold">{t.interpretation}</h2>
        <p className="mt-2 text-slate-700 dark:text-slate-300">{binomial.interpretation}</p>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.1fr_.9fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <h2 className="mb-4 text-lg font-semibold">{t.confusionMatrix}</h2>
          <div className="overflow-auto">
            <div className="grid min-w-[640px] gap-1" style={{ gridTemplateColumns: `72px repeat(${selectedLegacy.classes.length}, minmax(48px, 1fr))` }}>
              <div />
              {selectedLegacy.classes.map((className) => <div key={className} className="py-1 text-center text-xs font-semibold text-slate-500">{className}</div>)}
              {selectedLegacy.confusion_matrix.flatMap((row, rowIndex) => [
                <div key={`label-${rowIndex}`} className="flex items-center justify-end pr-2 text-xs font-semibold text-slate-500">{selectedLegacy.classes[rowIndex]}</div>,
                ...row.map((value, colIndex) => (
                  <div key={`${rowIndex}-${colIndex}`} title={`${selectedLegacy.classes[rowIndex]} -> ${selectedLegacy.classes[colIndex]}`} className="min-h-10 rounded-sm p-2 text-center text-xs font-semibold text-slate-950" style={{ backgroundColor: `rgba(20, 184, 166, ${0.1 + (value / matrixMax) * 0.8})` }}>
                    {value}
                  </div>
                )),
              ])}
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <h2 className="mb-4 text-lg font-semibold">{t.rocCurve}</h2>
          <svg viewBox="0 0 100 100" className="h-72 w-full rounded-md border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950" role="img" aria-label={t.rocCurve}>
            <line x1="0" y1="100" x2="100" y2="0" stroke="#94a3b8" strokeDasharray="4 4" strokeWidth="1.4" />
            <polyline points={rocPoints} fill="none" stroke="#0f766e" strokeWidth="2.5" />
          </svg>
          <div className="mt-4 rounded-md bg-slate-50 p-3 text-sm text-slate-700 dark:bg-slate-950 dark:text-slate-300">
            <h3 className="font-semibold">{t.modelInfo}</h3>
            <ul className="mt-2 space-y-1">{selectedLegacy.architecture.map((item) => <li key={item}>- {item}</li>)}</ul>
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <h2 className="mb-4 text-lg font-semibold">{t.rocComparison}</h2>
        <svg viewBox="0 0 100 100" className="h-80 w-full rounded-md border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950" role="img" aria-label={t.rocComparison}>
          <line x1="0" y1="100" x2="100" y2="0" stroke="#94a3b8" strokeDasharray="4 4" strokeWidth="1.2" />
          {legacy.roc_comparison.map((item, index) => (
            <polyline key={item.model_key} points={item.roc.fpr.map((fpr, pointIndex) => `${fpr * 100},${100 - item.roc.tpr[pointIndex] * 100}`).join(" ")} fill="none" stroke={palette[index % palette.length]} strokeWidth={item.model_key === selectedLegacy.model_key ? 2.8 : 1.8} />
          ))}
        </svg>
        <div className="mt-3 flex flex-wrap gap-3 text-sm">
          {legacy.roc_comparison.map((item, index) => (
            <span key={item.model_key} className="inline-flex items-center gap-2"><span className="h-3 w-3 rounded-full" style={{ backgroundColor: palette[index % palette.length] }} />{item.name} AUC {item.roc.auc.toFixed(2)}</span>
          ))}
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.1fr_.9fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <h2 className="mb-4 text-lg font-semibold">{t.modelComparison}</h2>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[680px] border-collapse text-sm">
              <thead><tr className="border-b border-slate-200 text-left text-slate-500 dark:border-slate-800 dark:text-slate-400"><th className="py-3 pr-4 font-semibold">{t.model}</th><th className="py-3 pr-4 font-semibold">{t.validationAccuracy}</th><th className="py-3 pr-4 font-semibold">{t.validationLoss}</th><th className="py-3 pr-4 font-semibold">{t.trainingTime}</th></tr></thead>
              <tbody>{legacy.model_comparison.map((row) => <tr key={row.model_key} className="border-b border-slate-100 dark:border-slate-800"><td className="py-3 pr-4 font-medium">{row.name}</td><td className="py-3 pr-4">{percent(row.validation_accuracy)}</td><td className="py-3 pr-4">{row.validation_loss.toFixed(4)}</td><td className="py-3 pr-4">{row.training_time_hours.toFixed(2)} h</td></tr>)}</tbody>
            </table>
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <h2 className="mb-4 text-lg font-semibold">{t.mcnemarTest}</h2>
          <div className="grid grid-cols-2 gap-3"><MetricBlock label="chi2" value={legacy.mcnemar_test.chi2.toFixed(4)} /><MetricBlock label="p-value" value={legacy.mcnemar_test.p_value === null ? "N/A" : legacy.mcnemar_test.p_value.toExponential(2)} detail={legacy.mcnemar_test.significant ? t.significant : t.notSignificant} tone={legacy.mcnemar_test.significant ? "strong" : "default"} /></div>
          <p className="mt-4 text-sm text-slate-700 dark:text-slate-300">{legacy.mcnemar_test.interpretation}</p>
          <h3 className="mb-2 mt-4 text-sm font-semibold">{t.mcnemarTable}</h3>
          <div className="overflow-x-auto"><table className="w-full min-w-[420px] border-collapse text-sm"><tbody>{legacy.mcnemar_test.table.map((row, rowIndex) => <tr key={rowIndex} className="border-b border-slate-100 dark:border-slate-800">{row.map((cell, cellIndex) => <td key={`${rowIndex}-${cellIndex}`} className="py-2 pr-3 font-medium text-slate-700 dark:text-slate-300">{cell}</td>)}</tr>)}</tbody></table></div>
        </div>
      </div>
    </section>
  );
}
function TrainingView({ t, training }: { t: typeof es; training: LegacyTrainingResponse | null }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">{t.trainingResults}</h2>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{training?.source ?? "legacy_precomputed_from_streamlit"}</p>
        </div>
        <BarChart3 className="text-teal-700 dark:text-teal-300" size={24} />
      </div>
      {training?.plot_available ? (
        <img src={`${api.url}${training.plot_endpoint}`} alt={t.trainingResults} className="max-h-[650px] w-full rounded-md border border-slate-200 object-contain dark:border-slate-800" />
      ) : (
        <EmptyPanel title={t.trainingResults} body={t.trainingPlotUnavailable} />
      )}
    </section>
  );
}

function DatasetView({ t, dataset }: { t: typeof es; dataset: LegacyDatasetResponse | null }) {
  return (
    <section className="grid gap-4 lg:grid-cols-[.8fr_1.2fr]">
      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <h2 className="text-lg font-semibold">{t.datasetInfo}</h2>
        <div className="mt-4 grid gap-3">
          <MetricBlock label={t.dataset} value={dataset?.dataset ?? "N/A"} detail={dataset?.source ?? "legacy_precomputed_from_streamlit"} />
          <MetricBlock label={t.images} value={dataset ? dataset.images.toLocaleString() : "N/A"} detail={t.histologyImages} />
          <MetricBlock label={t.classes} value={dataset ? String(dataset.classes) : "N/A"} detail={dataset?.resolution ?? "224x224"} />
        </div>
      </div>
      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <p className="text-sm leading-6 text-slate-700 dark:text-slate-300">{dataset?.summary ?? t.datasetUnavailable}</p>
        <div className="mt-5 flex flex-wrap gap-2">
          {dataset && Object.entries(dataset.links).map(([key, url]) => (
            <a key={key} href={url} target="_blank" rel="noreferrer" className="rounded-md border border-slate-300 px-3 py-2 text-sm font-medium hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800">
              {key}
            </a>
          ))}
        </div>
      </div>
    </section>
  );
}

function MetricBlock({ label, value, detail, tone = "default" }: { label: string; value: string; detail?: string; tone?: "default" | "strong" }) {
  return (
    <div className={`rounded-lg border p-4 ${tone === "strong" ? "border-teal-200 bg-teal-50 dark:border-teal-900 dark:bg-teal-950/50" : "border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900"}`}>
      <p className="text-xs font-semibold uppercase text-slate-500 dark:text-slate-400">{label}</p>
      <p className="mt-2 break-words text-2xl font-semibold tracking-normal text-slate-950 dark:text-slate-100">{value}</p>
      {detail && <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{detail}</p>}
    </div>
  );
}

function EmptyPanel({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-6 text-center shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <AlertCircle className="mx-auto text-slate-400" size={30} />
      <h2 className="mt-3 text-lg font-semibold">{title}</h2>
      <p className="mx-auto mt-2 max-w-xl text-sm text-slate-600 dark:text-slate-300">{body}</p>
    </div>
  );
}