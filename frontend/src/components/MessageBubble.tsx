import React from 'react';

// Later, we might add more specific types for sources if they have more structure
interface Source {
  text: string;
  score: number;
  // document_name?: string; // Example of more metadata
}

interface MessageBubbleProps {
  text: string;
  sender: 'user' | 'ai';
  isLoading?: boolean; // For AI messages, to show a typing/loading indicator
  sources?: Source[]; // For AI messages, to display source documents
  timestamp?: string; // Optional timestamp for the message
}

const MessageBubble: React.FC<MessageBubbleProps> = ({
  text,
  sender,
import SourceDocument from './SourceDocument'; // Import the SourceDocument component
import { MessageSource } from '../types/chat'; // Import MessageSource type

interface MessageBubbleProps {
  text: string;
  sender: 'user' | 'ai';
  isLoading?: boolean;
  sources?: MessageSource[]; // Updated to use MessageSource type
  timestamp?: Date;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({
  text,
  sender,
  isLoading = false,
  sources,
  timestamp,
}) => {
  const isUser = sender === 'user';

  // Base classes for all bubbles - increased padding slightly for better text breathing room
  const bubbleBaseClasses = "max-w-xl px-4 py-2.5 rounded-2xl shadow-md break-words";
  // User-specific classes - slightly darker blue, more distinct tail
  const userBubbleClasses = "bg-blue-600 dark:bg-blue-700 text-white ml-auto rounded-br-lg";
  // AI-specific classes - slightly lighter gray for dark mode, more distinct tail
  const aiBubbleClasses = "bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-gray-50 mr-auto rounded-bl-lg";

  const formatTimestamp = (date: Date | undefined): string => {
    if (!date) return '';
    return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', hour12: true });
  };

  return (
    <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} mb-3 w-full`}>
      <div
        className={`${bubbleBaseClasses} ${
          isUser ? userBubbleClasses : aiBubbleClasses
        }`}
        role="log" // Better accessibility for chat messages
        aria-live={isUser ? "off" : "polite"} // AI messages announce themselves when they arrive
        aria-atomic="false" // For aria-live, indicates that only changes should be announced
      >
        {isLoading && !text ? ( // Only show pure loading indicator if text is empty
          <div className="flex items-center space-x-1.5 py-1">
            <div className={`w-1.5 h-1.5 ${isUser ? 'bg-blue-200' : 'bg-gray-400 dark:bg-gray-500'} rounded-full animate-bounce [animation-delay:-0.3s]`}></div>
            <div className={`w-1.5 h-1.5 ${isUser ? 'bg-blue-200' : 'bg-gray-400 dark:bg-gray-500'} rounded-full animate-bounce [animation-delay:-0.15s]`}></div>
            <div className={`w-1.5 h-1.5 ${isUser ? 'bg-blue-200' : 'bg-gray-400 dark:bg-gray-500'} rounded-full animate-bounce`}></div>
          </div>
        ) : (
          // `text-sm sm:text-base` makes text slightly smaller on very small screens
          <p className="whitespace-pre-wrap text-sm sm:text-base leading-relaxed">{text}</p>
        )}
      </div>

      {/* Timestamp (Optional and not shown while AI is loading its initial empty response) */}
      {timestamp && (!isLoading || text) && (
        <p className={`text-xs mt-1.5 px-2 ${isUser ? 'text-right' : 'text-left'} text-gray-400 dark:text-gray-500 w-full`}>
          {formatTimestamp(timestamp)}
        </p>
      )}

      {/* Sources for AI messages (Optional) */}
      {!isUser && !isLoading && sources && sources.length > 0 && (
        // Reduced max-width for sources to be slightly inset from message bubble
        <div className="mt-2 w-full max-w-md self-start">
          <details className="group bg-gray-100 dark:bg-gray-700/50 p-2.5 rounded-lg border border-gray-200 dark:border-gray-600/60 shadow-sm">
            <summary className="cursor-pointer text-xs text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-gray-100 list-none flex items-center justify-between font-medium">
              <span>{sources.length} Source{sources.length > 1 ? 's' : ''}</span>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 transform transition-transform duration-200 group-open:rotate-180 text-gray-500 dark:text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </summary>
            <div className="mt-2 space-y-1.5 pt-2 border-t border-gray-200 dark:border-gray-600">
              {sources.map((source, index) => (
                <SourceDocument
                  key={index}
                  text={source.text}
                  score={source.score}
                  documentIndex={index + 1}
                />
              ))}
            </div>
          </details>
        </div>
      )}
    </div>
  );
};

export default MessageBubble;
