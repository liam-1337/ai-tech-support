/** @type {import('next').NextConfig} */
const nextConfig = {
  // reactStrictMode: true, // Default is true, good for development
  output: 'standalone', // Recommended for Electron packaging
  // experimental: {
  //   typedRoutes: true, // Optional: For type-safe routing
  // },

  // Example: If your backend API is on a different port during development
  // async rewrites() {
  //   return [
  //     {
  //       source: '/api/:path*',
  //       destination: 'http://localhost:8000/api/:path*', // Proxy to Backend
  //     },
  //   ]
  // },
};

export default nextConfig;
