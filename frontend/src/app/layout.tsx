import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import '../styles/globals.css'; // Import global styles
import Layout from '@/components/Layout'; // Import the main Layout component
import { ChatProvider } from '@/context/ChatContext'; // Import the ChatProvider

const inter = Inter({
  subsets: ['latin'],
  display: 'swap', // Ensures text remains visible during font loading
  variable: '--font-inter' // Optional: if you want to use it as a CSS variable
});

export const metadata: Metadata = {
  title: 'AI Tech Support Agent',
  description: 'AI Assistant for IT Support and Knowledge Retrieval',
  // Optional: Add more metadata like icons, open graph, etc.
  // icons: {
  //   icon: '/favicon.ico',
  // },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable || ''}>
      {/*
        Applying base background and text colors here for dark/light mode.
        The 'dark' class on <html> would be used if implementing a manual theme switcher.
        By default, Tailwind's dark: variants respond to OS preference.
      */}
      <body
        className={`${inter.className} antialiased bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors duration-300`}
      >
        <ChatProvider> {/* ChatProvider wraps everything inside the body */}
          <Layout> {/* Layout provides Header, Main, Footer structure */}
            {children} {/* children will be the page content (e.g., HomePage) */}
          </Layout>
        </ChatProvider>
      </body>
    </html>
  );
}
