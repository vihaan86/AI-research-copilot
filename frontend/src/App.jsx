import { useEffect, useMemo, useRef, useState } from 'react';

const API_BASE = '/api';

const initialMessage = {
  id: 'hello',
  role: 'assistant',
  content: 'Hi! How can I help you today?',
  sources: []
};

function normalizeDocument(document) {
  return {
    id: document.id,
    name: document.doc_name,
    path: document.file_path,
    indexedChunks: document.indexed_chunks || 0,
    indexingStatus: document.indexing_status || 'indexed',
    indexingError: document.indexing_error || ''
  };
}

function statusLabel(document) {
  if (document.indexingStatus === 'indexed') {
    return `${document.indexedChunks} chunks indexed`;
  }
  if (document.indexingStatus === 'failed') {
    return document.indexingError || 'Indexing failed';
  }
  return 'Indexing';
}

function App() {
  const [messages, setMessages] = useState([initialMessage]);
  const [question, setQuestion] = useState('');
  const [documents, setDocuments] = useState([]);
  const [uploadingNames, setUploadingNames] = useState([]);
  const [isChatting, setIsChatting] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [notice, setNotice] = useState('');
  const transcriptRef = useRef(null);
  const fileInputRef = useRef(null);

  const hasIndexingDocuments = useMemo(
    () => documents.some((document) => document.indexingStatus === 'indexing'),
    [documents]
  );

  useEffect(() => {
    refreshDocuments();
  }, []);

  useEffect(() => {
    if (!hasIndexingDocuments) return;

    const intervalId = window.setInterval(refreshDocuments, 2200);
    return () => window.clearInterval(intervalId);
  }, [hasIndexingDocuments]);

  useEffect(() => {
    transcriptRef.current?.scrollTo({
      top: transcriptRef.current.scrollHeight,
      behavior: 'smooth'
    });
  }, [messages, isChatting]);

  async function refreshDocuments() {
    try {
      const response = await fetch(`${API_BASE}/documents`);
      if (!response.ok) return;

      const payload = await response.json();
      setDocuments(payload.map(normalizeDocument));
    } catch {
      setNotice('Could not refresh uploaded PDFs.');
    }
  }

  async function uploadFiles(fileList) {
    const files = Array.from(fileList || []).filter(
      (file) => file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
    );

    if (!files.length) {
      setNotice('Choose one or more PDF files.');
      return;
    }

    setNotice('');
    setUploadingNames(files.map((file) => file.name));

    for (const file of files) {
      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch(`${API_BASE}/documents`, {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          const errorPayload = await response.json().catch(() => ({}));
          throw new Error(errorPayload.detail || 'Upload failed');
        }

        const payload = await response.json();
        setDocuments((prev) => [normalizeDocument(payload), ...prev]);
      } catch (error) {
        setNotice(`${file.name}: ${error.message || 'Upload failed'}`);
      } finally {
        setUploadingNames((prev) => prev.filter((name) => name !== file.name));
      }
    }
  }

  async function removeDocument(documentId) {
    setDocuments((prev) => prev.filter((document) => document.id !== documentId));

    try {
      const response = await fetch(`${API_BASE}/documents/${documentId}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        const errorPayload = await response.json().catch(() => ({}));
        throw new Error(errorPayload.detail || 'Remove failed');
      }
    } catch (error) {
      setNotice(error.message || 'Could not remove PDF.');
      refreshDocuments();
    }
  }

  async function sendMessage() {
    const content = question.trim();
    if (!content || isChatting) return;

    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      sources: []
    };

    setMessages((prev) => [...prev, userMessage]);
    setQuestion('');
    setIsChatting(true);
    setNotice('');

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: content, top_k: 5 })
      });

      if (!response.ok) {
        const errorPayload = await response.json().catch(() => ({}));
        throw new Error(errorPayload.detail || 'Chat request failed');
      }

      const payload = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: payload.answer || 'I do not have enough information to answer that.',
          sources: payload.sources || []
        }
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: error.message || 'Something went wrong while contacting the assistant.',
          sources: []
        }
      ]);
    } finally {
      setIsChatting(false);
    }
  }

  function handleComposerKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    uploadFiles(event.dataTransfer.files);
  }

  const sidebar = (
    <aside className="upload-sidebar" aria-label="PDF upload panel">
      <div className="sidebar-heading">
        <div>
          <p className="eyebrow">Sources</p>
          <h2>PDFs</h2>
        </div>
        <button className="panel-close" type="button" onClick={() => setIsSidebarOpen(false)}>
          Close
        </button>
      </div>

      <button
        className={`drop-zone ${isDragging ? 'dragging' : ''}`}
        type="button"
        onClick={() => fileInputRef.current?.click()}
        onDragEnter={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <span className="drop-mark">PDF</span>
        <span className="drop-title">Drop PDFs here</span>
        <span className="drop-copy">or click to browse</span>
      </button>

      <input
        ref={fileInputRef}
        type="file"
        accept="application/pdf,.pdf"
        multiple
        hidden
        onChange={(event) => {
          uploadFiles(event.target.files);
          event.target.value = '';
        }}
      />

      {notice ? <p className="notice">{notice}</p> : null}

      <div className="file-list" aria-live="polite">
        {uploadingNames.map((name) => (
          <div className="file-row uploading" key={`uploading-${name}`}>
            <div>
              <p className="file-name">{name}</p>
              <p className="file-meta">Uploading</p>
            </div>
            <span className="mini-progress" />
          </div>
        ))}

        {documents.map((document) => (
          <div className="file-row" key={document.id}>
            <div>
              <p className="file-name">{document.name}</p>
              <p className={`file-meta ${document.indexingStatus}`}>{statusLabel(document)}</p>
            </div>
            <button
              className="remove-file"
              type="button"
              aria-label={`Remove ${document.name}`}
              onClick={() => removeDocument(document.id)}
            >
              Remove
            </button>
          </div>
        ))}

        {!uploadingNames.length && !documents.length ? (
          <p className="empty-state">No PDFs uploaded yet.</p>
        ) : null}
      </div>
    </aside>
  );

  return (
    <div className="app-shell">
      <main className="chat-panel">
        <header className="topbar">
          <div>
            <p className="eyebrow">Research workspace</p>
            <h1>AI Assistant</h1>
          </div>
          <button className="sources-toggle" type="button" onClick={() => setIsSidebarOpen(true)}>
            PDFs
          </button>
        </header>

        <section className="transcript" ref={transcriptRef} aria-live="polite">
          {messages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              <div className="message-label">{message.role === 'assistant' ? 'AI Assistant' : 'You'}</div>
              <div className="bubble">
                <p>{message.content}</p>
                {message.sources?.length ? (
                  <div className="references">
                    <div className="references-label">References</div>
                    <div className="source-strip">
                    {message.sources.slice(0, 3).map((source, index) => (
                      <span key={`${source.document_name}-${source.page_number}-${index}`}>
                        {source.document_name}, p. {source.page_number}
                      </span>
                    ))}
                    </div>
                  </div>
                ) : null}
              </div>
            </article>
          ))}

          {isChatting ? (
            <article className="message assistant entering">
              <div className="message-label">AI Assistant</div>
              <div className="bubble thinking" aria-label="Assistant is thinking">
                <span />
                <span />
                <span />
              </div>
            </article>
          ) : null}
        </section>

        <footer className="composer-wrap">
          <div className="composer">
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              placeholder="Ask anything about your PDFs..."
              rows={1}
            />
            <button type="button" onClick={sendMessage} disabled={!question.trim() || isChatting}>
              Send
            </button>
          </div>
        </footer>
      </main>

      {sidebar}

      <div className={`mobile-panel ${isSidebarOpen ? 'open' : ''}`}>
        <button className="panel-scrim" type="button" aria-label="Close PDF panel" onClick={() => setIsSidebarOpen(false)} />
        {sidebar}
      </div>
    </div>
  );
}

export default App;
