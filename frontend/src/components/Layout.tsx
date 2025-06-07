import React from 'react';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    // The base bg/text colors are now set on the <body> tag in RootLayout.
    // This div can focus on its flex structure.
    <div className="min-h-screen flex flex-col">
      <header className="bg-gradient-to-r from-blue-600 to-indigo-700 dark:from-blue-700 dark:to-indigo-900 text-white py-4 px-4 sm:px-6 lg:px-8 shadow-md sticky top-0 z-50">
        <div className="container mx-auto flex items-center justify-between max-w-7xl">
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight">
            AI Tech Support Agent
          </h1>
          {/* Placeholder for potential future elements like user profile or theme toggle */}
          <div className="flex items-center space-x-2">
            {/* Example:
            <button
              aria-label="Toggle theme"
              className="p-2 rounded-full hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-white/50"
            >
              ☀️ / 🌙
            </button> */}
          </div>
        </div>
      </header>

      {/* The main content area grows to fill available space. */}
      {/* max-w-7xl ensures content width is capped for very wide screens, matching header. */}
      <main className="flex-grow w-full container mx-auto p-4 sm:p-6 lg:p-8 max-w-7xl">
        {children}
      </main>

      <footer className="w-full text-center p-4 sm:p-5 text-xs sm:text-sm text-gray-500 dark:text-gray-400 border-t border-gray-200 dark:border-gray-700">
        &copy; {new Date().getFullYear()} AI Tech Support Solutions.
        {/* Optional: Add links or other footer content here */}
        {/* <span className="mx-1">|</span>
          <a href="/privacy" className="hover:underline">Privacy Policy</a> |
          <a href="/terms" className="hover:underline">Terms of Service</a>
        </p> */}
      </footer>
    </div>
  );
};

export default Layout;
