import { useState, useEffect, useRef, useCallback } from "react";
import "./App.css";

const BACKEND_WS = "ws://localhost:8000/query";
const BACKEND_HTTP = "http://localhost:8000";

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [citations, setCitations] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [wsStatus, setWsStatus] = useState("Connecting...");
  const [streamTime, setStreamTime] = useState(null);

  const socketRef = useRef(null);
  const responseRef = useRef(null);

  const connectWebSocket = useCallback(() => {
    const ws = new WebSocket(BACKEND_WS);

    ws.onopen = () => {
      setWsStatus("Connected");
      console.log("[WS] Connected");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        switch (data.type) {
          case "token":
            setResponse((prev) => prev + data.payload);
            if (responseRef.current) {
              responseRef.current.scrollTop = responseRef.current.scrollHeight;
            }
            break;

          case "citation":
            setCitations((prev) => [...prev, data.payload]);
            break;

          case "done":
            setIsStreaming(false);
            if (data.payload?.total_time) {
              setStreamTime(data.payload.total_time);
            }
            break;

          case "error":
            setIsStreaming(false);
            alert("Error: " + data.payload);
            break;

          default:
            console.warn("[WS] Unknown message type:", data.type);
        }
      } catch (err) {
        console.error("[WS] Failed to parse message:", err);
      }
    };

    ws.onerror = () => {
      setWsStatus("Error");
      console.error("[WS] Connection error");
    };

    ws.onclose = () => {
      setWsStatus("Disconnected");
      console.log("[WS] Disconnected - reconnecting in 3s...");
      setTimeout(connectWebSocket, 3000);
    };

    socketRef.current = ws;
  }, []);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (socketRef.current) socketRef.current.close();
    };
  }, [connectWebSocket]);

  const sendQuery = () => {
    if (!query.trim()) return;
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      alert("WebSocket is not connected. Please wait...");
      return;
    }
    setResponse("");
    setCitations([]);
    setStreamTime(null);
    setIsStreaming(true);
    socketRef.current.send(query);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !isStreaming) sendQuery();
  };

  const uploadFile = async () => {
    if (!file) {
      setUploadStatus("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    setUploadStatus("Uploading...");

    try {
      const res = await fetch(`${BACKEND_HTTP}/ingest`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setUploadStatus(
        `${data.message || "Upload successful"} - ${data.filename || file.name}`
      );
    } catch (err) {
      setUploadStatus("Upload failed: " + err.message);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-title">
          <h1>Real-Time Streaming RAG</h1>
          <p className="subtitle">High-precision document retrieval & streaming response synthesis</p>
        </div>
        <span className={`ws-badge ${wsStatus === "Connected" ? "connected" : "disconnected"}`}>
          {wsStatus}
        </span>
      </header>

      <div className="main-grid">
        <section className="panel query-panel">
          <h2>Ask a Question</h2>
          <div className="input-row">
            <input
              type="text"
              placeholder="e.g., What degree is Lalitha pursuing and at which college?"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isStreaming}
            />
            <button
              onClick={sendQuery}
              disabled={isStreaming || !query.trim()}
              className="primary-btn"
            >
              {isStreaming ? "Streaming..." : "Ask"}
            </button>
          </div>

          <h3>
            Answer
            {isStreaming && <span className="streaming-dot"> ●</span>}
          </h3>
          <div className="response-box" ref={responseRef}>
            {response || (
              <span className="placeholder">Response will stream here in real time...</span>
            )}
          </div>

          {(streamTime !== null || citations.length > 0) && (
            <div className="retrieval-summary">
              <span>Retrieved {citations.length} relevant chunk{citations.length === 1 ? "" : "s"}</span>
              {streamTime && <span> • Response generated in {streamTime}s</span>}
            </div>
          )}

          {citations.length > 0 && (
            <div className="sources-container">
              <h3>Sources & Citations</h3>
              <ul className="citation-list">
                {citations.map((c, i) => (
                  <li key={i} className="citation-item">
                    <div className="citation-header">
                      <span className="source-title">📄 {c.source || "Document"}</span>
                      {c.section && <span className="badge section-badge">{c.section}</span>}
                      {c.title && c.title !== c.section && (
                        <span className="badge title-badge">{c.title}</span>
                      )}
                      {c.page !== null && c.page !== undefined && (
                        <span className="badge">Page {c.page}</span>
                      )}
                      {c.relevance_score !== undefined && (
                        <span className="badge score">
                          Relevance: {Math.round(c.relevance_score * 100)}%
                        </span>
                      )}
                    </div>
                    {c.snippet && (
                      <p className="snippet">"{c.snippet}"</p>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>

        <section className="panel upload-panel">
          <h2>Ingest Document</h2>
          <p className="hint">
            Upload a <code>.txt</code> or <code>.pdf</code> file. It is
            parsed with section-aware chunking and indexed asynchronously in ChromaDB.
          </p>
          <div className="upload-row">
            <input
              type="file"
              accept=".txt,.pdf,.md,.csv"
              onChange={(e) => {
                setFile(e.target.files[0]);
                setUploadStatus("");
              }}
            />
            <button onClick={uploadFile} className="secondary-btn">
              Upload
            </button>
          </div>
          {uploadStatus && <p className="upload-status">{uploadStatus}</p>}
        </section>
      </div>
    </div>
  );
}

export default App;