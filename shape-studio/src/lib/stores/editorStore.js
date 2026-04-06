import { writable, get } from 'svelte/store';
import { deepCopyState, generatePreset, validateShape } from '../utils/geometry.js';

const init = {
    pts: generatePreset('circle'),
    history: [], historyIdx: -1,
    selectedIdx: -1,
    showHighRes: false, normalizeExport: false,
    reference: { points: null, visible: false }, // ✅ Hidden by default
    toast: { msg: '', type: 'success', visible: false }
};

const store = writable(init);

function save() {
    store.update(s => {
        let h = [...s.history];
        if (s.historyIdx < h.length - 1) h = h.slice(0, s.historyIdx + 1);
        h.push(deepCopyState(s.pts));
        if (h.length > 50) h.shift();
        return { ...s, history: h, historyIdx: h.length - 1 };
    });
}

export const shapeState = store;
export const actions = {
    setPts(newPts) { save(); store.update(s => ({ ...s, pts: newPts, selectedIdx: -1 })); },
    loadPreset(type) { actions.setPts(generatePreset(type)); },

    updatePointLocal(idx, x, y) {
        store.update(s => ({ ...s, pts: s.pts.map((p, i) => i === idx ? { ...p, x, y } : p) }));
    },
    commitDrag(idx, x, y) {
        store.update(s => ({ ...s, pts: s.pts.map((p, i) => i === idx ? { ...p, x, y } : p) }));
        save();
    },
    updatePoint(idx, axis, val) {
        const n = parseFloat(val); if (isNaN(n)) return;
        store.update(s => ({ ...s, pts: s.pts.map((p, i) => i === idx ? { ...p, [axis]: n } : p) }));
        save();
    },
    undo() {
        store.update(s => {
            if (s.historyIdx <= 0) return s;
            return { ...s, pts: s.history[s.historyIdx - 1].map(p => ({ ...p })), historyIdx: s.historyIdx - 1 };
        });
    },
    redo() {
        store.update(s => {
            if (s.historyIdx >= s.history.length - 1) return s;
            return { ...s, pts: s.history[s.historyIdx + 1].map(p => ({ ...p })), historyIdx: s.historyIdx + 1 };
        });
    },
    selectPoint(idx) { store.update(s => ({ ...s, selectedIdx: idx })); },
    toggleHighRes() { store.update(s => ({ ...s, showHighRes: !s.showHighRes })); },
    toggleNormalize() { store.update(s => ({ ...s, normalizeExport: !s.normalizeExport })); },

    loadReference(file) {
        if (!file) return;
        const r = new FileReader();
        r.onload = e => {
            try {
                const json = JSON.parse(e.target.result);
                // Explicitly handle your exported format: { meta: {...}, points: [[x,y], ...] }
                const raw = json.points || json;
                if (!Array.isArray(raw) || raw.length < 2) throw new Error("Invalid or empty points array");

                const cleaned = raw.map(p => {
                    if (Array.isArray(p) && p.length >= 2) return { x: p[0], y: p[1] };
                    if (p && typeof p === 'object') return { x: p.x || 0, y: p.y || 0 };
                    return { x: 0, y: 0 };
                });

                store.update(s => ({ ...s, reference: { points: cleaned, visible: true } }));
                actions.fireToast(`📐 Reference loaded (${cleaned.length} pts)`);
            } catch (err) { actions.fireToast("❌ Ref import failed: " + err.message, "error"); }
        };
        r.readAsText(file);
    },
    toggleReference() {
        store.update(s => ({ ...s, reference: { ...s.reference, visible: !s.reference.visible } }));
    },
    clearReference() {
        store.update(s => ({ ...s, reference: { points: null, visible: false } }));
        actions.fireToast("🗑️ Reference cleared");
    },
    fireToast(msg, type = 'success') {
        store.update(s => ({ ...s, toast: { msg, type, visible: true } }));
        setTimeout(() => store.update(s => ({ ...s, toast: { ...s.toast, visible: false } })), 2500);
    },
    handleImport(file) {
        if (!file) return;
        const r = new FileReader();
        r.onload = e => {
            try {
                const j = JSON.parse(e.target.result);
                const pts = j.points || j;
                if (!Array.isArray(pts) || pts.length !== 32) throw new Error("Expected exactly 32 points");
                const newPts = pts.map((p, i) => {
                    if (!Array.isArray(p) || p.length !== 2) throw new Error(`Point ${i} must be [x, y]`);
                    return { x: parseFloat(p[0]) || 0, y: parseFloat(p[1]) || 0, anchor: i % 2 === 0 };
                });
                actions.setPts(newPts);
                store.update(s => ({ ...s, history: [], historyIdx: -1 }));
                save();
                actions.fireToast("✅ Shape imported successfully");
            } catch (err) { actions.fireToast("❌ Import failed: " + err.message, "error"); }
        };
        r.readAsText(file);
    },
    downloadJSON() {
        const s = get(store);
        let data = s.pts.map(p => [parseFloat(p.x.toFixed(4)), parseFloat(p.y.toFixed(4))]);
        if (s.normalizeExport) {
            const m = Math.max(...data.flat().map(Math.abs)) || 1;
            data = data.map(([x, y]) => [x / m, y / m]);
        }
        const payload = { meta: { version: "1.0", type: "quad_bezier", segments: 16, normalized: s.normalizeExport, winding: validateShape(s.pts).winding, timestamp: Date.now() }, points: data };
        const b = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
        const u = URL.createObjectURL(b);
        const a = document.createElement("a"); a.href = u; a.download = `shape_32pt_v${Date.now().toString().slice(-4)}.json`;
        document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(u);
        actions.fireToast("✅ Exported successfully");
    }
};
