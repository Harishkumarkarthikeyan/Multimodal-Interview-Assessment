import React, { useEffect, useMemo, useRef, useState } from "react";

const EMPTY_REPORT = {
  scores: {},
  metrics: {},
  strengths: [],
  improvements: [],
  transcript: "",
};

const NAV_ITEMS = [
  { id: "overview", label: "Overview" },
  { id: "analyze", label: "Analyze" },
  { id: "scores", label: "Scores" },
  { id: "feedback", label: "Feedback" },
  { id: "transcript", label: "Transcript" },
  { id: "architecture", label: "Architecture" },
];

const SCORE_CARDS = [
  {
    key: "confidence",
    label: "Confidence",
    detail: "Voice energy, face visibility, steadiness, and composed movement.",
  },
  {
    key: "communication",
    label: "Communication",
    detail: "Pace, filler words, answer depth, and sentence clarity.",
  },
  {
    key: "non_verbal_presence",
    label: "Non-Verbal Presence",
    detail: "Face detection, posture visibility, and visual stability.",
  },
  {
    key: "employability",
    label: "Employability",
    detail: "Combined readiness score across all interview signals.",
  },
];

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ||
  (window.location.port === "5173" ? "http://127.0.0.1:8000" : "");

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

function scoreLabel(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "--";
  return Math.round(Number(value));
}

function percent(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "--";
  return `${Math.round(Number(value) * 100)}%`;
}

function ScoreRing({ value, size = "large" }) {
  const score = Math.max(0, Math.min(100, Number(value) || 0));
  return (
    <div className={`score-ring ${size}`} style={{ "--score": score }}>
      <span>{scoreLabel(value)}</span>
    </div>
  );
}

function FeedbackList({ items, fallback }) {
  const values = items?.length ? items : [fallback];
  return (
    <ul className={`feedback-list ${items?.length ? "" : "muted-list"}`}>
      {values.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PipelineStep({ index, title, tool, description }) {
  return (
    <article className="pipeline-step">
      <span>{index}</span>
      <div>
        <h3>{title}</h3>
        <strong>{tool}</strong>
        <p>{description}</p>
      </div>
    </article>
  );
}

export default function App() {
  const [activePage, setActivePage] = useState("overview");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [serverOnline, setServerOnline] = useState(false);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [job, setJob] = useState({ status: "idle", message: "Waiting for a video" });
  const [report, setReport] = useState(EMPTY_REPORT);
  const [isDragging, setIsDragging] = useState(false);
  const pollRef = useRef(null);
  const inputRef = useRef(null);

  const transcriptWordCount = useMemo(() => {
    const text = report.transcript?.trim();
    return text ? text.split(/\s+/).length : 0;
  }, [report.transcript]);

  const hasReport = Boolean(report.transcript || Object.keys(report.scores || {}).length);
  const metrics = report.metrics || {};
  const employability = report.scores?.employability;

  useEffect(() => {
    fetch(apiUrl("/api/health"))
      .then((response) => {
        if (!response.ok) throw new Error("Server offline");
        setServerOnline(true);
      })
      .catch(() => setServerOnline(false));
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [previewUrl]);

  function navigate(page) {
    setActivePage(page);
    setDrawerOpen(false);
  }

  function chooseFile(nextFile) {
    if (!nextFile) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(nextFile);
    setPreviewUrl(URL.createObjectURL(nextFile));
    setJob({ status: "ready", message: `Ready: ${nextFile.name}` });
    setActivePage("analyze");
  }

  async function loadDemo() {
    if (pollRef.current) clearInterval(pollRef.current);
    setJob({ status: "complete", message: "Demo assessment loaded" });
    const response = await fetch(apiUrl("/api/demo-report"));
    setReport(await response.json());
    setActivePage("scores");
  }

  async function startAnalysis() {
    if (!file) return;
    if (pollRef.current) clearInterval(pollRef.current);

    setJob({ status: "processing", message: "Uploading video" });
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(apiUrl("/api/analyze"), {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Upload failed");
      setJob({ status: "processing", message: "Analysis started" });
      pollJob(data.job_id);
    } catch (error) {
      setJob({ status: "failed", message: error.message });
    }
  }

  function pollJob(jobId) {
    pollRef.current = setInterval(async () => {
      try {
        const response = await fetch(apiUrl(`/api/jobs/${jobId}`));
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Could not read job status");

        setJob({ status: data.status, message: data.message || data.status });
        if (data.status === "complete") {
          clearInterval(pollRef.current);
          setReport(data.report);
          setActivePage("scores");
        }
        if (data.status === "failed") {
          clearInterval(pollRef.current);
        }
      } catch (error) {
        clearInterval(pollRef.current);
        setJob({ status: "failed", message: error.message });
      }
    }, 1800);
  }

  function downloadReport() {
    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json",
    });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "interview_assessment.json";
    link.click();
    URL.revokeObjectURL(link.href);
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <button
          className="menu-button"
          type="button"
          aria-label="Open navigation"
          onClick={() => setDrawerOpen(true)}
        >
          <span />
          <span />
          <span />
        </button>
        <div className="brand-inline">
          <div className="brand-mark">MI</div>
          <div>
            <h1>Interview Assessment</h1>
            <p>Multimodal AI Evaluation</p>
          </div>
        </div>
        <div className={`status-pill ${serverOnline ? "ok" : "bad"}`}>
          <span className="status-dot" />
          {serverOnline ? "Server ready" : "Server offline"}
        </div>
      </header>

      <aside className={`drawer ${drawerOpen ? "open" : ""}`} aria-hidden={!drawerOpen}>
        <div className="drawer-top">
          <div className="brand-inline">
            <div className="brand-mark">MI</div>
            <div>
              <h2>Interview Assessment</h2>
              <p>Project Navigation</p>
            </div>
          </div>
          <button className="close-button" type="button" onClick={() => setDrawerOpen(false)}>
            x
          </button>
        </div>

        <nav className="drawer-nav">
          {NAV_ITEMS.map((item) => (
            <button
              className={activePage === item.id ? "active" : ""}
              type="button"
              key={item.id}
              onClick={() => navigate(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <section className="model-card" aria-label="Pipeline stack">
          <span className="eyebrow">Pipeline</span>
          <div className="pipeline-row"><span>Video</span><strong>MediaPipe</strong></div>
          <div className="pipeline-row"><span>Audio</span><strong>Librosa</strong></div>
          <div className="pipeline-row"><span>Text</span><strong>Whisper</strong></div>
          <div className="pipeline-row"><span>Backend</span><strong>FastAPI</strong></div>
        </section>
      </aside>

      {drawerOpen ? <button className="drawer-scrim" type="button" aria-label="Close navigation" onClick={() => setDrawerOpen(false)} /> : null}

      <section className="workspace">
        {activePage === "overview" ? (
          <section className="page-grid overview-page">
            <article className="hero-panel">
              <span className="eyebrow">Candidate Review</span>
              <h2>AI-Powered Multimodal Interview Performance Assessment</h2>
              <p>
                Upload an interview recording, analyze speech and visual cues, then present
                explainable feedback through a polished candidate dashboard.
              </p>
              <div className="hero-actions">
                <button className="primary-btn" type="button" onClick={() => navigate("analyze")}>
                  Start Assessment
                </button>
                <button className="ghost-btn" type="button" onClick={loadDemo}>
                  Load Demo
                </button>
              </div>
            </article>
            <article className="summary-panel">
              <ScoreRing value={employability} />
              <h3>{hasReport ? "Latest Employability Score" : "Ready For Presentation"}</h3>
              <p>
                {hasReport
                  ? "Review the generated report from the Scores, Feedback, and Transcript pages."
                  : "Use demo mode for viva, then switch to real video processing when the ML stack is installed."}
              </p>
            </article>
          </section>
        ) : null}

        {activePage === "analyze" ? (
          <section className="page-grid analyze-page">
            <article className="upload-panel">
              <div
                className={`drop-zone ${isDragging ? "dragging" : ""}`}
                onDragEnter={(event) => {
                  event.preventDefault();
                  setIsDragging(true);
                }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={(event) => {
                  event.preventDefault();
                  setIsDragging(false);
                }}
                onDrop={(event) => {
                  event.preventDefault();
                  setIsDragging(false);
                  chooseFile(event.dataTransfer.files[0]);
                }}
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept="video/*"
                  onChange={(event) => chooseFile(event.target.files[0])}
                />
                <div className="upload-icon">+</div>
                <h3>Upload interview video</h3>
                <p>MP4, MOV, AVI, MKV, or WEBM</p>
                <button className="secondary-btn" type="button" onClick={() => inputRef.current.click()}>
                  Choose Video
                </button>
              </div>
              <div className="action-row">
                <button
                  className="primary-btn"
                  type="button"
                  disabled={!file || job.status === "processing"}
                  onClick={startAnalysis}
                >
                  {job.status === "processing" ? "Analyzing..." : "Analyze Video"}
                </button>
                <button className="ghost-btn" type="button" onClick={loadDemo}>
                  Load Demo
                </button>
              </div>
              <div className={`job-status ${job.status}`}>
                <span className="small-dot" />
                {job.message}
              </div>
            </article>

            <article className={`preview-panel ${previewUrl ? "has-video" : ""}`}>
              {previewUrl ? <video src={previewUrl} controls playsInline /> : null}
              {!previewUrl ? (
                <div className="empty-preview">
                  <div className="frame-lines" />
                  <p>Video preview appears here</p>
                </div>
              ) : null}
            </article>
          </section>
        ) : null}

        {activePage === "scores" ? (
          <section className="stack-page">
            <div className="page-title">
              <span className="eyebrow">Assessment</span>
              <h2>Score Dashboard</h2>
            </div>
            <div className="score-grid">
              {SCORE_CARDS.map((card) => (
                <article className="score-card" key={card.key}>
                  <ScoreRing value={report.scores?.[card.key]} size="medium" />
                  <div>
                    <h3>{card.label}</h3>
                    <p>{card.detail}</p>
                  </div>
                </article>
              ))}
            </div>
            <div className="metric-grid">
              <Metric label="Duration" value={metrics.duration_seconds ? `${Math.round(metrics.duration_seconds)}s` : "--"} />
              <Metric label="Words/min" value={metrics.words_per_minute ? Math.round(metrics.words_per_minute) : "--"} />
              <Metric label="Filler words" value={metrics.filler_words ?? "--"} />
              <Metric label="Face visibility" value={percent(metrics.face_visibility)} />
              <Metric label="Sampled frames" value={metrics.sampled_frames ?? "--"} />
              <Metric label="Voice energy" value={metrics.voice_energy ?? "--"} />
            </div>
          </section>
        ) : null}

        {activePage === "feedback" ? (
          <section className="stack-page">
            <div className="page-title split">
              <div>
                <span className="eyebrow">Feedback</span>
                <h2>Strengths And Improvements</h2>
              </div>
              <button className="icon-btn" type="button" disabled={!hasReport} onClick={downloadReport}>
                Download JSON
              </button>
            </div>
            <div className="feedback-columns">
              <article className="panel">
                <h3>Strengths</h3>
                <FeedbackList items={report.strengths} fallback="No assessment loaded yet." />
              </article>
              <article className="panel">
                <h3>Improvements</h3>
                <FeedbackList items={report.improvements} fallback="Upload a video or load the demo report." />
              </article>
            </div>
          </section>
        ) : null}

        {activePage === "transcript" ? (
          <section className="stack-page">
            <div className="page-title split">
              <div>
                <span className="eyebrow">Transcript</span>
                <h2>Speech-To-Text Output</h2>
              </div>
              <span className="word-count">{transcriptWordCount} words</span>
            </div>
            <article className="panel transcript-panel">
              <p>{report.transcript || "The generated transcript will appear here after analysis."}</p>
            </article>
          </section>
        ) : null}

        {activePage === "architecture" ? (
          <section className="stack-page">
            <div className="page-title">
              <span className="eyebrow">System Design</span>
              <h2>Multimodal Pipeline Architecture</h2>
            </div>
            <div className="pipeline-grid">
              <PipelineStep index="01" title="Video Input" tool="OpenCV" description="Reads uploaded interview recordings and samples frames for visual analysis." />
              <PipelineStep index="02" title="Visual Cues" tool="MediaPipe" description="Extracts face, pose, and hand landmarks to estimate presence and stability." />
              <PipelineStep index="03" title="Speech Features" tool="Librosa" description="Measures audio energy, duration, tempo, and speech-friendly acoustic signals." />
              <PipelineStep index="04" title="Transcript" tool="Whisper" description="Converts spoken answers into text for communication and content scoring." />
              <PipelineStep index="05" title="Assessment" tool="FastAPI + Python" description="Combines metrics into explainable scores, feedback, and downloadable reports." />
            </div>
          </section>
        ) : null}
      </section>
    </main>
  );
}
