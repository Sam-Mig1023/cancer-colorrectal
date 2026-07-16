"use client";

import { type FormEvent, type ReactNode, useEffect, useRef, useState } from "react";
import {
  Bot,
  LoaderCircle,
  MessageCircle,
  Mic,
  MicOff,
  Send,
  User,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import type { ModelInfo, PredictionResponse } from "@/types/api";
import es from "@/messages/es.json";
import { api } from "@/services/api";

type Language = "es" | "en";
type TabKey = "diagnosis" | "analysis" | "training" | "dataset";
type ChatMessage = { id: number; role: "assistant" | "user"; text: string };

interface SpeechRecognitionEventLike {
  results: ArrayLike<{ 0: { transcript: string }; isFinal: boolean }>;
}

interface SpeechRecognitionErrorEventLike {
  error: string;
}

interface SpeechRecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start: () => void;
  stop: () => void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  }
}

function inlineMarkdown(text: string): ReactNode[] {
  return text.split(/(\*\*[^*\n]+\*\*)/g).filter(Boolean).map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={`${part}-${index}`} className="font-semibold text-slate-950 dark:text-white">{part.slice(2, -2)}</strong>;
    }
    return <span key={`${part}-${index}`}>{part}</span>;
  });
}

function FormattedAssistantMessage({ text }: { text: string }) {
  const normalized = text
    .replace(/\s+(?=\d+\.\s+)/g, "\n")
    .replace(/\s+(?=[-•]\s+)/g, "\n");
  const lines = normalized.split(/\n+/).map((line) => line.trim()).filter(Boolean);
  const blocks: Array<{ kind: "paragraph" | "ordered" | "unordered"; items: string[] }> = [];

  for (const rawLine of lines) {
    const ordered = rawLine.match(/^\d+\.\s+(.+)$/);
    const unordered = rawLine.match(/^[-•]\s+(.+)$/);
    const kind = ordered ? "ordered" : unordered ? "unordered" : "paragraph";
    const content = ordered?.[1] ?? unordered?.[1] ?? rawLine.replace(/^#{1,3}\s+/, "");
    const previous = blocks.at(-1);
    if (previous && previous.kind === kind && kind !== "paragraph") {
      previous.items.push(content);
    } else {
      blocks.push({ kind, items: [content] });
    }
  }

  return (
    <div className="space-y-2.5">
      {blocks.map((block, blockIndex) => {
        if (block.kind === "ordered") {
          return <ol key={blockIndex} className="list-decimal space-y-2 pl-5 marker:font-semibold marker:text-teal-700 dark:marker:text-teal-300">{block.items.map((item, index) => <li key={index} className="pl-1">{inlineMarkdown(item)}</li>)}</ol>;
        }
        if (block.kind === "unordered") {
          return <ul key={blockIndex} className="list-disc space-y-2 pl-5 marker:text-teal-700 dark:marker:text-teal-300">{block.items.map((item, index) => <li key={index} className="pl-1">{inlineMarkdown(item)}</li>)}</ul>;
        }
        return <p key={blockIndex}>{inlineMarkdown(block.items[0])}</p>;
      })}
    </div>
  );
}

export default function HelpChatbot({
  language,
  t,
  activeTab,
  selectedModel,
  result,
}: {
  language: Language;
  t: typeof es;
  activeTab: TabKey;
  selectedModel?: ModelInfo;
  result: PredictionResponse | null;
}) {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [listening, setListening] = useState(false);
  const [loading, setLoading] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: 1, role: "assistant", text: t.chatGreeting },
  ]);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const nextId = useRef(2);

  const speechRecognitionAvailable = typeof window !== "undefined" && Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
  const speechSynthesisAvailable = typeof window !== "undefined" && "speechSynthesis" in window;

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, open]);

  useEffect(() => {
    setMessages((current) => {
      if (current.length === 1 && current[0].role === "assistant") {
        return [{ ...current[0], text: t.chatGreeting }];
      }
      return current;
    });
  }, [t.chatGreeting]);

  useEffect(() => () => {
    recognitionRef.current?.stop();
    window.speechSynthesis?.cancel();
  }, []);

  const speak = (text: string) => {
    if (!voiceEnabled || !speechSynthesisAvailable) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = language === "es" ? "es-PE" : "en-US";
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
  };

  const sendMessage = async (rawText: string) => {
    const text = rawText.trim();
    if (!text || loading) return;
    const userMessage: ChatMessage = { id: nextId.current++, role: "user", text };
    const conversation = [...messages, userMessage];
    setMessages(conversation);
    setInput("");
    setLoading(true);
    try {
      const response = await api.chat({
        language,
        messages: conversation.slice(-12).map((message) => ({ role: message.role, content: message.text })),
        context: {
          active_tab: activeTab,
          selected_model: selectedModel?.name,
          predicted_class: result?.predicted_class,
          confidence: result?.confidence,
        },
      });
      setMessages((current) => [...current, { id: nextId.current++, role: "assistant", text: response.answer }]);
      speak(response.answer);
    } catch (error) {
      const code = error instanceof Error ? error.message : "";
      const errorText = code === "API_503"
        ? t.chatNotConfigured
        : code === "API_401"
          ? t.chatInvalidCredentials
          : code === "API_403"
            ? t.chatModelNotAllowed
        : code === "API_429"
          ? t.chatRateLimit
          : t.chatApiError;
      setMessages((current) => [...current, { id: nextId.current++, role: "assistant", text: errorText }]);
    } finally {
      setLoading(false);
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void sendMessage(input);
  };

  const toggleListening = () => {
    if (listening) {
      recognitionRef.current?.stop();
      setListening(false);
      return;
    }

    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) return;
    const recognition = new Recognition();
    recognition.lang = language === "es" ? "es-PE" : "en-US";
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.onresult = (event) => {
      let transcript = "";
      let isFinal = false;
      for (let index = 0; index < event.results.length; index += 1) {
        transcript += event.results[index][0].transcript;
        isFinal ||= event.results[index].isFinal;
      }
      setInput(transcript);
      if (isFinal) void sendMessage(transcript);
    };
    recognition.onerror = () => {
      setListening(false);
      setMessages((current) => [...current, { id: nextId.current++, role: "assistant", text: t.chatMicError }]);
    };
    recognition.onend = () => setListening(false);
    recognitionRef.current = recognition;
    setListening(true);
    recognition.start();
  };

  const toggleVoice = () => {
    const next = !voiceEnabled;
    setVoiceEnabled(next);
    if (!next) window.speechSynthesis?.cancel();
  };

  return (
    <div className="fixed bottom-4 right-4 z-50 sm:bottom-6 sm:right-6">
      {open && (
        <section
          role="dialog"
          aria-label={t.chatTitle}
          className="mb-3 flex h-[min(620px,calc(100vh-7rem))] w-[calc(100vw-2rem)] max-w-sm flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-slate-900"
        >
          <header className="flex items-center gap-3 bg-teal-700 px-4 py-3 text-white">
            <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-white/15"><Bot size={22} /></span>
            <div className="min-w-0 flex-1">
              <h2 className="truncate font-semibold">{t.chatTitle}</h2>
              <p className="text-xs text-teal-50">{t.chatOnline}</p>
            </div>
            <button
              type="button"
              onClick={toggleVoice}
              disabled={!speechSynthesisAvailable}
              className="grid h-9 w-9 place-items-center rounded-full hover:bg-white/15 disabled:opacity-40"
              title={voiceEnabled ? t.chatDisableVoice : t.chatEnableVoice}
              aria-label={voiceEnabled ? t.chatDisableVoice : t.chatEnableVoice}
            >
              {voiceEnabled ? <Volume2 size={19} /> : <VolumeX size={19} />}
            </button>
            <button type="button" onClick={() => setOpen(false)} className="grid h-9 w-9 place-items-center rounded-full hover:bg-white/15" aria-label={t.chatClose}>
              <X size={20} />
            </button>
          </header>

          <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto bg-slate-50 p-4 dark:bg-slate-950" aria-live="polite">
            {messages.map((message) => (
              <div key={message.id} className={`flex items-end gap-2 ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                {message.role === "assistant" && <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-teal-100 text-teal-800 dark:bg-teal-950 dark:text-teal-200"><Bot size={15} /></span>}
                <div className={`max-w-[86%] rounded-2xl px-3.5 py-3 text-sm leading-6 ${message.role === "user" ? "rounded-br-sm bg-teal-700 text-white" : "rounded-bl-sm border border-slate-200 bg-white text-slate-800 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"}`}>
                  {message.role === "assistant" ? <FormattedAssistantMessage text={message.text} /> : message.text}
                </div>
                {message.role === "user" && <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-200"><User size={15} /></span>}
              </div>
            ))}
            {loading && (
              <div className="flex items-end gap-2">
                <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-teal-100 text-teal-800 dark:bg-teal-950 dark:text-teal-200"><Bot size={15} /></span>
                <div className="flex items-center gap-2 rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300">
                  <LoaderCircle className="animate-spin" size={16} /> {t.chatThinking}
                </div>
              </div>
            )}
          </div>

          <div className="border-t border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-900">
            <div className="mb-2 flex gap-2 overflow-x-auto pb-1">
              {t.chatQuickQuestions.map((question) => (
                <button key={question} type="button" disabled={loading} onClick={() => void sendMessage(question)} className="shrink-0 rounded-full border border-teal-200 bg-teal-50 px-3 py-1.5 text-xs font-medium text-teal-900 hover:bg-teal-100 disabled:opacity-50 dark:border-teal-900 dark:bg-teal-950 dark:text-teal-100 dark:hover:bg-teal-900">
                  {question}
                </button>
              ))}
            </div>
            <form onSubmit={submit} className="flex items-center gap-2">
              <label htmlFor="chat-message" className="sr-only">{t.chatPlaceholder}</label>
              <input
                id="chat-message"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                disabled={loading}
                placeholder={listening ? t.chatListening : t.chatPlaceholder}
                className="h-11 min-w-0 flex-1 rounded-full border border-slate-300 bg-white px-4 text-sm dark:border-slate-700 dark:bg-slate-950"
              />
              <button
                type="button"
                onClick={toggleListening}
                disabled={!speechRecognitionAvailable}
                className={`grid h-11 w-11 shrink-0 place-items-center rounded-full border disabled:cursor-not-allowed disabled:opacity-40 ${listening ? "border-rose-600 bg-rose-600 text-white animate-pulse" : "border-slate-300 text-slate-700 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800"}`}
                title={speechRecognitionAvailable ? (listening ? t.chatStopListening : t.chatStartListening) : t.chatVoiceUnavailable}
                aria-label={speechRecognitionAvailable ? (listening ? t.chatStopListening : t.chatStartListening) : t.chatVoiceUnavailable}
              >
                {listening ? <MicOff size={19} /> : <Mic size={19} />}
              </button>
              <button type="submit" disabled={!input.trim() || loading} className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-teal-700 text-white hover:bg-teal-800 disabled:cursor-not-allowed disabled:bg-slate-400" aria-label={t.chatSend}>
                <Send size={18} />
              </button>
            </form>
            <p className="mt-2 text-center text-[11px] text-slate-500 dark:text-slate-400">{t.chatDisclaimer}</p>
          </div>
        </section>
      )}

      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
        aria-label={open ? t.chatClose : t.chatOpen}
        className="ml-auto flex h-14 items-center gap-2 rounded-full bg-teal-700 px-4 font-semibold text-white shadow-lg transition hover:bg-teal-800 hover:shadow-xl"
      >
        {open ? <X size={22} /> : <MessageCircle size={22} />}
        <span>{open ? t.chatClose : t.chatButton}</span>
      </button>
    </div>
  );
}
