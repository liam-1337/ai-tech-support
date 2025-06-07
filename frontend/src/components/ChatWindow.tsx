import React from 'react';

import React, { useState, useEffect, useRef, FormEvent } from 'react';
import { useChat } from '../context/ChatContext';
import { fetchQueryResponse } from '../utils/apiClient';
import { ChatMessage as ChatMessageType, MessageSource } from '../types/chat'; // Renamed to avoid conflict
import MessageBubble from './MessageBubble';

const ChatWindow: React.FC = () => {
  const {
    messages,
    addMessage,
    updateMessage, // Added from context
    setIsLoadingResponse,
    setError,
    isLoadingResponse, // Global loading state from context
    error,
  } = useChat();

  const [inputValue, setInputValue] = useState<string>('');
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmedInput = inputValue.trim();
    if (!trimmedInput || isLoadingResponse) return; // Prevent sending if loading or empty

    setError(null);
    addMessage({ text: trimmedInput, sender: 'user' });

    const placeholderAiMsg = addMessage({
      text: '',
      sender: 'ai',
      isLoading: true,
    });

    setInputValue('');
    setIsLoadingResponse(true);

    try {
      const apiResponse = await fetchQueryResponse({ question: trimmedInput });
      updateMessage(placeholderAiMsg.id, {
        text: apiResponse.generated_answer,
        sources: apiResponse.source_chunks.map(s => ({ text: s.text, score: s.score })),
        isLoading: false,
      });
      if (apiResponse.message) {
        console.info("Backend message:", apiResponse.message);
      }
    } catch (apiError: any) {
      const errorText = apiError.message || 'Sorry, I encountered an error getting a response.';
      updateMessage(placeholderAiMsg.id, {
        text: errorText,
        isLoading: false,
      });
      setError(errorText);
    } finally {
      setIsLoadingResponse(false);
    }
  };

  // Adjusted height calculation: Considers typical header/footer heights.
  // The exact subtractions might need tweaking based on final Layout.tsx header/footer height.
  // Example: header (py-4 ~64px) + footer (p-4 ~56px) + page padding (py-4/8 ~32px) = ~152px.
  // Let's use a slightly more generous subtraction to allow for some page padding.
  const chatAreaHeight = "h-[calc(100vh-16rem)] sm:h-[calc(100vh-14rem)]"; // Approx 256px/224px total for chrome

  return (
    <div className={`flex flex-col ${chatAreaHeight} max-w-3xl w-full mx-auto bg-white dark:bg-gray-800 shadow-2xl rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700`}>
      {/* Error Display Area */}
      {error && (
        <div className="p-3 bg-red-100 dark:bg-red-900/60 border-b border-red-300 dark:border-red-700 text-red-700 dark:text-red-200 text-sm font-medium">
          <span className="font-semibold">Error:</span> {error}
        </div>
      )}

      {/* Message Display Area */}
      {/* Added more padding for messages and a subtle background pattern */}
      <div className="flex-grow p-4 sm:p-6 space-y-2 overflow-y-auto scrolling-touch bg-gray-50/50 dark:bg-gray-800/50">
        {messages.length === 0 && !isLoadingResponse && !error && (
          <div className="flex flex-col items-center justify-center h-full text-gray-400 dark:text-gray-500 text-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
            <p className="text-lg">Conversation starts here.</p>
            <p className="text-sm">Ask any IT-related question to get started.</p>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            text={msg.text}
            sender={msg.sender}
            isLoading={msg.isLoading}
            sources={msg.sources}
            timestamp={msg.timestamp}
          />
        ))}
        {/* Empty div to ensure scrolling to the bottom works */}
        <div ref={messagesEndRef} />
      </div>

      {/* Message Input Area */}
      <div className={`bg-gray-50 dark:bg-gray-700 p-3 sm:p-4 border-t border-gray-200 dark:border-gray-600 ${isLoadingResponse ? 'opacity-70 pointer-events-none' : ''}`}>
        <form onSubmit={handleSubmit} className="flex items-center space-x-2 sm:space-x-3">
          <input
            type="text"
            name="messageInput"
            placeholder="Type your question or issue here..."
            aria-label="Chat message input"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            disabled={isLoadingResponse} // Disable input while AI is thinking globally
            className="flex-grow p-3 border border-gray-300 dark:border-gray-500 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none dark:bg-gray-600 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 disabled:opacity-60"
          />
          <button
            type="submit"
            aria-label="Send message"
            disabled={isLoadingResponse || !inputValue.trim()} // Disable if loading or input is empty
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-5 sm:px-6 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-700 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {isLoadingResponse ? (
              <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 sm:h-6 sm:w-6" viewBox="0 0 20 20" fill="currentColor">
              <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 16.571V11.5a1 1 0 011-1h.094a1 1 0 01.994.89l.812 5.22A1 1 0 0013 17.571l5 1.428a1 1 0 001.17-1.408l-7-14z" />
            </svg>
            )}
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatWindow;
