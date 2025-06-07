import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}', // If using Pages Router
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}', // For App Router
  ],
  theme: {
    extend: {
      // Add custom theme extensions here
      // Example:
      // colors: {
      //   'brand-primary': '#0070f3',
      // },
      // fontFamily: {
      //   sans: ['var(--font-inter)', 'sans-serif'],
      // },
    },
  },
  plugins: [],
}
export default config
