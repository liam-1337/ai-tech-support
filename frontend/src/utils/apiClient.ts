// frontend/src/utils/apiClient.ts

// --- API Data Types/Interfaces ---

export interface QueryApiRequest {
  question: string;
  top_k?: number;
}

export interface SourceChunkApiResponse {
  text: string;
  score: number;
  // Potential future fields:
  // source_document_name?: string;
  // page_number?: number;
  // original_chunk_id?: string;
}

export interface QueryApiResponse {
  generated_answer: string;
  source_chunks: SourceChunkApiResponse[];
  message?: string; // Optional messages from backend, e.g., warnings or info
}

// Interface for structured error responses from the API (e.g., FastAPI HTTPException)
export interface ApiErrorDetail {
  msg: string;
  type: string;
  loc?: (string | number)[]; // Location of the error, e.g., in request body
}
export interface ApiErrorResponse {
  detail?: string | ApiErrorDetail[]; // FastAPI uses 'detail' for HTTPExceptions
}

// --- API Client Function ---

/**
 * Fetches a response from the backend AI Tech Support Agent API.
 *
 * @param request The query request object containing the question and optional top_k.
 * @returns A promise that resolves to the API response.
 * @throws An error if the API call fails or returns a non-ok status.
 */
export async function fetchQueryResponse(request: QueryApiRequest): Promise<QueryApiResponse> {
  // Determine API base URL:
  // NEXT_PUBLIC_ prefix is essential for Next.js to expose it to the browser.
  const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

  // Align with the actual FastAPI endpoint (currently /query/)
  const endpoint = `${apiBaseUrl}/query/`;

  console.log(`Fetching from endpoint: ${endpoint} with request:`, request); // For debugging

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json', // Good practice to specify accept header
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      let errorMessage = `API Error: ${response.status} ${response.statusText}`;
      try {
        // Attempt to parse a structured error response from the backend
        const errorData: ApiErrorResponse = await response.json();
        if (typeof errorData.detail === 'string') {
          errorMessage = errorData.detail;
        } else if (Array.isArray(errorData.detail)) { // Handle FastAPI validation errors
          errorMessage = errorData.detail
            .map(err => `${err.loc ? err.loc.join(' -> ') + ': ' : ''}${err.msg} (type: ${err.type})`)
            .join('; ');
        }
      } catch (e) {
        // If parsing errorData fails, stick with the original HTTP status message
        console.warn('Could not parse error response as JSON:', e);
      }
      console.error('API request failed:', errorMessage);
      throw new Error(errorMessage);
    }

    // If response is OK, parse the JSON
    const data: QueryApiResponse = await response.json();
    return data;

  } catch (error) {
    // Handles network errors, or errors thrown from the !response.ok block
    console.error('Failed to fetch query response due to an exception:', error);

    if (error instanceof Error) {
      // Re-throw the error, possibly a more user-friendly one or one that can be better handled by UI
      throw new Error(`Network or application error: ${error.message}`);
    }
    // Fallback for unknown error types
    throw new Error('An unexpected error occurred while communicating with the API.');
  }
}

// Example usage (for testing within this file or in components):
/*
async function testApi() {
  try {
    const response = await fetchQueryResponse({ question: "How to reset password?", top_k: 3 });
    console.log("API Response:", response);
    // response.generated_answer
    // response.source_chunks.forEach(chunk => console.log(chunk.text, chunk.score));
  } catch (error) {
    if (error instanceof Error) {
      console.error("Test API Error:", error.message);
    } else {
      console.error("Test API Error (unknown type):", error);
    }
  }
}

// To test, uncomment and call testApi() in a useEffect or similar in a component.
// Make sure the backend FastAPI server is running.
*/
