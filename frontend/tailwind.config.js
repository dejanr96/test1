/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'apex-dark': '#0f172a',
                'apex-panel': '#1e293b',
                'apex-green': '#10b981',
                'apex-red': '#ef4444',
                'apex-blue': '#3b82f6',
            }
        },
    },
    plugins: [],
}
