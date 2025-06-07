'use client';

import React, { useState, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import { MessageSource } from '../types/chat'; // Import the MessageSource type

// Define the expected structure of the response from /demo_query/
interface DemoQueryResponseData {
  question: string;
  generated_answer: string;
  source_chunks: MessageSource[];
  message?: string;
}

const DemoChatWindow: React.FC = () => {
  const [demoData, setDemoData] = useState<DemoQueryResponseData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDemoData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch('/demo_query/'); // FastAPI backend is at root
        if (!response.ok) {
          throw new Error(`Failed to fetch demo data: ${response.status} ${response.statusText}`);
        }
        const data: DemoQueryResponseData = await response.json();
        setDemoData(data);
      } catch (err: any) {
        setError(err.message || 'An unknown error occurred while fetching demo data.');
        console.error("Fetch demo data error:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchDemoData();
  }, []); // Empty dependency array ensures this runs only once on mount

  // Adjusted height to be more flexible, not full screen chat window height
  const chatAreaHeight = "min-h-[300px]";

  return (
    <div className={`flex flex-col ${chatAreaHeight} max-w-3xl w-full mx-auto bg-white dark:bg-gray-800 shadow-2xl rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700`}>
      {/* Error Display Area */}
      {error && !isLoading && (
        <div className="p-3 bg-red-100 dark:bg-red-900/60 border-b border-red-300 dark:border-red-700 text-red-700 dark:text-red-200 text-sm font-medium">
          <span className="font-semibold">Error:</span> {error}
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="flex flex-col items-center justify-center flex-grow p-6 text-gray-500 dark:text-gray-400">
          <svg className="animate-spin h-10 w-10 mb-3 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <p className="text-lg">Loading Demo Content...</p>
        </div>
      )}

      {/* Message Display Area */}
      {!isLoading && !error && demoData && (
        <div className="flex-grow p-4 sm:p-6 space-y-4 overflow-y-auto scrolling-touch bg-gray-50/50 dark:bg-gray-800/50">
          {/* Display the Question */}
          <MessageBubble
            text={demoData.question}
            sender="user"
            timestamp={new Date()} // Add a mock timestamp
          />

          {/* Display the Answer and Sources */}
          <MessageBubble
            text={demoData.generated_answer}
            sender="ai"
            sources={demoData.source_chunks}
            timestamp={new Date(new Date().getTime() + 1000)} // Mock timestamp slightly after question
          />

          {demoData.message && (
            <p className="text-xs text-center text-gray-500 dark:text-gray-400 p-2">{demoData.message}</p>
          )}
        </div>
      )}
       {!isLoading && !error && !demoData && (
         <div className="flex flex-col items-center justify-center flex-grow p-6 text-gray-500 dark:text-gray-400">
            <p className="text-lg">No demo data loaded.</p>
          </div>
       )}
    </div>
  );
};

export default DemoChatWindow;
