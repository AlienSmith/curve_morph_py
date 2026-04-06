import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  server: {
    port: 5173, // Change this to 5174 if 5173 is still busy
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000', // Use 127.0.0.1 for more reliable local routing
        changeOrigin: true,
        secure: false,
        // rewrite: (path) => path.replace(/^\/api/, '') 
        // ^ Only uncomment the line above if your Python FastAPI 
        //   routes DON'T start with /api (e.g., @app.post("/generate-morph"))
      }
    }
  }
})