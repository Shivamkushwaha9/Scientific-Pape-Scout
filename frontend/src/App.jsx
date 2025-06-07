import React, { useState, useRef, useEffect } from 'react';
import { Search, FileText, Download, Clock } from 'lucide-react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [toolLog, setToolLog] = useState([]);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // In a real implementation, this would connect to your FastAPI backend
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      let assistantMessage = { 
        role: 'assistant', 
        content: '', 
        timestamp: new Date(),
        streaming: true 
      };
      
      setMessages(prev => [...prev, assistantMessage]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            
            if (data.type === 'content') {
              setMessages(prev => {
                const newMessages = [...prev];
                const lastMessage = newMessages[newMessages.length - 1];
                lastMessage.content += data.content;
                return newMessages;
              });
            } else if (data.type === 'tool_call') {
              setToolLog(prev => [...prev, data]);
            }
          }
        }
      }

      setMessages(prev => {
        const newMessages = [...prev];
        const lastMessage = newMessages[newMessages.length - 1];
        lastMessage.streaming = false;
        return newMessages;
      });

    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
        error: true
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>🔬 Scientific Paper Scout</h1>
          <p>Discover and summarize research papers with AI</p>
        </div>
      </header>

      <div className="main-container">
        <div className="chat-section">
          <div className="messages">
            {messages.map((message, index) => (
              <div key={index} className={`message ${message.role}`}>
                <div className="message-content">
                  {message.content}
                  {message.streaming && <span className="cursor">|</span>}
                </div>
                <div className="message-time">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-section">
            <div className="input-container">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                placeholder="Ask about research papers..."
                disabled={isLoading}
              />
              <button 
                onClick={sendMessage} 
                disabled={isLoading || !input.trim()}
                className="send-button"
              >
                <Search size={20} />
              </button>
            </div>
          </div>
        </div>

        <div className="sidebar">
          <div className="tool-log">
            <h3>🔧 Tool Calls</h3>
            {toolLog.length === 0 ? (
              <p className="no-tools">No tool calls yet</p>
            ) : (
              <div className="tool-entries">
                {toolLog.slice(-5).map((entry, index) => (
                  <div key={index} className="tool-entry">
                    <div className="tool-name">
                      {entry.server}.{entry.tool}
                    </div>
                    <div className="tool-time">
                      <Clock size={12} />
                      {entry.latency}ms
                    </div>
                    <div className={`tool-status ${entry.success ? 'success' : 'error'}`}>
                      {entry.success ? '✅' : '❌'}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;