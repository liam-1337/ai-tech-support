// frontend/src/context/ChatContext.tsx
import React, {
  createContext,
  useContext,
  useState,
  ReactNode,
  useCallback,
  useMemo, // Can be used to memoize context value
} from 'react';
import { ChatMessage, ChatContextType, MessageSource, MessageSender } from '../types/chat'; // Adjust path as needed

// 1. Create the Context
const ChatContext = createContext<ChatContextType | undefined>(undefined);

// 2. Create the Provider Component
interface ChatProviderProps {
  children: ReactNode;
}

export const ChatProvider: React.FC<ChatProviderProps> = ({ children }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoadingResponse, setIsLoadingResponse] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const addMessage = useCallback((messageContent: Omit<ChatMessage, 'id' | 'timestamp'>): ChatMessage => {
    const newMessage: ChatMessage = {
      ...messageContent,
      id: crypto.randomUUID(),
      timestamp: new Date(),
    };
    setMessages(prevMessages => [...prevMessages, newMessage]);
    return newMessage;
  }, []);

  const updateMessage = useCallback((id: string, updates: Partial<Omit<ChatMessage, 'id' | 'timestamp'>>) => {
    setMessages(prevMessages =>
      prevMessages.map(msg =>
        msg.id === id ? { ...msg, ...updates, isLoading: updates.isLoading ?? msg.isLoading } : msg
      )
    );
  }, []);

  // Other state update functions are straightforward setters
  const handleSetMessages = useCallback((newMessages: ChatMessage[]) => {
    setMessages(newMessages);
  }, []);

  const handleSetIsLoadingResponse = useCallback((isLoading: boolean) => {
    setIsLoadingResponse(isLoading);
  }, []);

  const handleSetError = useCallback((newError: string | null) => {
    setError(newError);
  }, []);

  // The value provided to consuming components by the context.
  // Consider using useMemo here if contextValue objects were complex and frequently recomputed,
  // but for now, direct object creation is fine.
  // useMemo to stabilize the context value, preventing unnecessary re-renders of consumers
  // if the provider's parent re-renders but these state values/functions haven't changed.
  const contextValue = useMemo(() => ({
    messages,
    setMessages: handleSetMessages,
    isLoadingResponse,
    setIsLoadingResponse: handleSetIsLoadingResponse,
    error,
    setError: handleSetError,
    addMessage,
    updateMessage, // Add the new updateMessage function to the context value
  }), [messages, isLoadingResponse, error, handleSetMessages, handleSetIsLoadingResponse, handleSetError, addMessage, updateMessage]);

  return (
    <ChatContext.Provider value={contextValue}>
      {children}
    </ChatContext.Provider>
  );
};

// 3. Create a Custom Hook for using the Context
/**
 * Custom hook to access the ChatContext.
 * Provides a convenient way to use chat state and actions in components.
 * Throws an error if used outside of a ChatProvider.
 */
export const useChat = (): ChatContextType => {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};
