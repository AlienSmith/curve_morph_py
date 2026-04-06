import { writable } from 'svelte/store';

// --- EDITOR STORES ---

// The "Source of Truth" for your 32 vector points
export const points = writable(Array.from({ length: 32 }, (_, i) => ({
    x: Math.cos((i / 32) * Math.PI * 2) * 0.5,
    y: Math.sin((i / 32) * Math.PI * 2) * 0.5
})));

// The URL for the reference image being traced in the background
export const bgImage = writable(null);

// Global toggle for the grid visibility
export const showGrid = writable(true);


// --- PREVIEWER STORES (For your Morphing Logic) ---

export const shapeA = writable(null); // File object
export const shapeB = writable(null); // File object
export const morphResult = writable(null); // Blob URL for the GIF
export const renderDpi = writable(100);
export const status = writable("Ready");