import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy the API in dev so the browser sees a single origin. This sidesteps
    // CORS entirely locally; the backend's CORS config is what covers a
    // deployment where the two are served from different hosts.
    proxy: {
      // 8010 rather than the usual 8000, which is commonly already taken.
      // Change here and in the uvicorn --port flag together.
      '/api': {
        target: 'http://127.0.0.1:8010',
        changeOrigin: true,
      },
    },
  },
})
