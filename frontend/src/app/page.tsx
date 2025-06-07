import ChatWindow from '@/components/ChatWindow'; // Import the ChatWindow component

// Home Page for the AI Tech Support Agent Frontend
export default function HomePage() {
  return (
    // The main Layout component (from src/app/layout.tsx via src/components/Layout.tsx)
    // already provides header, footer, and overall page structure.
    // This HomePage component will render its content within the <main> of that Layout.
    <section className="w-full flex flex-col items-center justify-start py-4 md:py-8">
      {/*
        The <header> and <footer> are now part of the global Layout.tsx.
        This page.tsx should focus on the content specific to this page.
        For the main page, it's primarily the ChatWindow.
      */}
      <ChatWindow />
    </section>
  )
}
