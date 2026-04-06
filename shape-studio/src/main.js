import './app.css' // We'll create a simple global CSS next
import 'bootstrap/dist/css/bootstrap.min.css'
import App from './App.svelte'

const app = new App({
  target: document.getElementById('app'),
})

export default app