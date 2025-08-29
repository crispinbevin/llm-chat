import { useState } from "react";
import ReactMarkdown from 'react-markdown';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [persona, setPersona] = useState("chatbot");
  const [loading, setLoading] = useState(false);

  const parseMarkdown = (text: str) => {
  let parsed = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  
  parsed = parsed.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>');
  
  parsed = parsed.replace(/\n/g, '<br>');
  
  return parsed;
  };
  

  const sendMessage = async () => {
    if (!input.trim()) return;
    setLoading(true);

    // add user message
    const newMessages = [...messages, { role: "user", content: input }];
    setMessages(newMessages);

    // send to backend
    const res = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ persona, message: input }),
    });
    const data = await res.json();

    // add assistant reply
    setMessages([...newMessages, { role: "assistant", content: data.reply }]);
    setInput("");
    setLoading(false);
  };

  return (
    <div className="flex flex-col h-screen bg-zinc-900 text-white">
      <div className="div flex items-center justify-between mx-12 mt-12">
        <h1 className="text-5xl font-bold  border-bottom-1">
          LLM Chat
        </h1>
        <button className="bg-green-500 p-3 rounded-lg hover:cursor-pointer" onClick={ () => window.location.reload()}>New Chat +</button>
      </div>
      {/* Chat area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 mt-12">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${
              msg.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-[70%] px-4 py-2 rounded-2xl ${
                msg.role === "user"
                  ? "bg-green-600 text-zinc-50"
                  : "bg-zinc-700 text-zinc-200"
              }`}
            >
              {msg.role === "user" ? (
    msg.content
  ) : (
    <div dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.content) }} />
  )}
            </div>
          </div>
        ))}
        {loading && (
      <div className="flex justify-end">
        <div className="bg-zinc-700 text-zinc-100 max-w-[70%] px-4 py-2 rounded-2xl flex items-center gap-2">
          <div className="animate-spin h-4 w-4 border-2 border-zinc-400 border-t-transparent rounded-full"></div>
          <span className="text-sm">Thinking...</span>
        </div>
      </div>
    )}
      </div>

      {/* Input area */}
      <div className="p-4 border-t border-zinc-700 flex gap-2">
        <select
          value={persona}
          onChange={(e) => setPersona(e.target.value)}
          className="bg-zinc-800 text-white rounded-lg px-3 py-2"
        >
          <option value="email">Concise Email Writer</option>
          <option value="teacher">Explanatory Teacher</option>
          <option value="technical">Technical Explainer</option>
          <option value="chatbot">Normal Chatbot</option>
        </select>
        
        <input
          type="text"
          className="flex-1 bg-zinc-800 text-white rounded-lg px-3 py-2 focus:outline-none"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button
          onClick={sendMessage}
          className="bg-green-400 px-4 py-2 rounded-lg hover:bg-green-600 "
          disabled={loading}
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default App;
