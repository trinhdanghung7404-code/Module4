import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Port 5174 is pinned on purpose. The backend CORS allow-list
// (app.cors.allowed-origins in application.properties) only accepts 5173 and
// 5174. Without strictPort, Vite silently moves to 5175 when something else
// holds 5174 and then every /api/** response gets blocked by the browser with
// a CORS error that looks like a backend bug.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    strictPort: true,
  },
})
