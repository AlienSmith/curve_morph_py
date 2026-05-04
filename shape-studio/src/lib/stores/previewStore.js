import { writable } from 'svelte/store';

// --- EDITOR STORES ---
export const points = writable(Array.from({ length: 32 }, (_, i) => ({
    x: Math.cos((i / 32) * Math.PI * 2) * 0.5,
    y: Math.sin((i / 32) * Math.PI * 2) * 0.5
})));
export const bgImage = writable(null);
export const showGrid = writable(true);

// --- PREVIEWER STORES ---
export const shapeA = writable(null);
export const shapeB = writable(null);
export const morphResult = writable(null); // GIF blob URL
export const morphBinary = writable(null); // NEW: .morph file blob URL
export const renderDpi = writable(100);
export const status = writable("Ready");