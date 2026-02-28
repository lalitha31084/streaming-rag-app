import { useState, useEffect } from "react";

function App() {
  const [socket, setSocket] = useState(null);
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [citations, setCitations] = useState([]);
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/query");

    ws.onopen = () => {
      console.log("Connected to WebSocket");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "token") {
        setResponse((prev) => prev + data.payload);
      }

      if (data.type === "citation") {
        setCitations(data.payload);
      }

      if (data.type === "error") {
        alert("Error: " + data.payload);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    setSocket(ws);

    return () => {
      ws.close();
    };
  }, []);

  const sendQuery = () => {
    setResponse("");
    setCitations([]);
    socket.send(query);
  };

  const uploadFile = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setStatus("Uploading...");

    const res = await fetch("http://localhost:8000/ingest", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setStatus(data.message);
  };

  return (
    <div style={{ padding: "40px", fontFamily: "Arial" }}>
      <h2>🚀 Real-Time Streaming RAG</h2>

      <div>
        <input
          type="text"
          placeholder="Ask a question..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ width: "400px", padding: "8px" }}
        />
        <button onClick={sendQuery} style={{ marginLeft: "10px" }}>
          Ask
        </button>
      </div>

      <h3>Answer:</h3>
      <div style={{ border: "1px solid gray", padding: "10px", minHeight: "100px" }}>
        {response}
      </div>

      <h3>Citations:</h3>
      <ul>
        {citations.map((c, index) => (
          <li key={index}>{c}</li>
        ))}
      </ul>

      <hr />

      <h3>Upload Document</h3>
      <input
        type="file"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button onClick={uploadFile} style={{ marginLeft: "10px" }}>
        Upload
      </button>

      <p>{status}</p>
    </div>
  );
}

export default App;