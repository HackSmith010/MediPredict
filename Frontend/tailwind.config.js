/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: '#4A90E2',
        secondary: '#008080',
        background: '#F5F7FA',
        success: '#28A745',
        error: '#D9534F',
        heading: '#2C3E50',
        body: '#6C757D',
      },
    },
  },
  plugins: [],
};