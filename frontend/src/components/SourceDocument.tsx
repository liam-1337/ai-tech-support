import React from 'react';

interface SourceDocumentProps {
  text: string;
  score: number;
  sourceName?: string; // e.g., filename or document title
  documentIndex?: number; // e.g., "Source 1", "Source 2"
  // onSelect?: () => void; // Optional: if clicking the source should do something
}

const SourceDocument: React.FC<SourceDocumentProps> = ({
  text,
  score,
  sourceName,
  documentIndex,
  // onSelect,
}) => {
  const displayTitle =
    documentIndex !== undefined ? `Source ${documentIndex}`
    : sourceName ? sourceName
    : 'Context Snippet';

  return (
    // Using 'group' class on details allows styling based on open/closed state if needed elsewhere
    <details className="block bg-gray-100 dark:bg-gray-700/60 p-2.5 rounded-md border border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500 transition-all duration-150 ease-in-out">
      <summary className="cursor-pointer text-xs font-semibold text-gray-700 dark:text-gray-300 list-none flex justify-between items-center group-hover:text-gray-900 dark:group-hover:text-gray-100">
        <span className="truncate" title={sourceName || `Source ${documentIndex}`}>
          {displayTitle}
          <span className="ml-1.5 text-gray-500 dark:text-gray-400 font-normal">(Score: {score.toFixed(3)})</span>
        </span>
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-3.5 w-3.5 text-gray-500 dark:text-gray-400 transform transition-transform duration-200 group-open:rotate-180"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2.5} // Slightly thicker arrow
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </summary>
      {/* Content revealed when <details> is open */}
      <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-500/70">
        <p className="text-xs text-gray-600 dark:text-gray-300 whitespace-pre-wrap leading-snug">
          {text}
        </p>
      </div>
    </details>
  );
};

export default SourceDocument;
