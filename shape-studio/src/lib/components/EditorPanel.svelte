<script>
    import { X } from "lucide-svelte";
    import { shapeState, actions } from "../stores/editorStore.js";
    import { validateShape } from "../utils/geometry.js";

    $: ({ pts, selectedIdx } = $shapeState);
    $: winding = selectedIdx !== -1 ? validateShape(pts).winding : null;
</script>

{#if selectedIdx !== -1}
    <div
        class="props-panel position-absolute top-0 end-0 m-2 p-2 bg-dark border border-secondary rounded shadow"
    >
        <div class="d-flex justify-content-between align-items-center mb-2">
            <strong class="small">Point #{selectedIdx}</strong>
            <button
                class="btn-close btn-close-white"
                style="font-size:0.5rem;"
                on:click={() => actions.selectPoint(-1)}><X size={12} /></button
            >
        </div>
        <div class="mb-1">
            <label class="form-label small m-0 text-secondary">X</label>
            <input
                type="number"
                class="form-control form-control-sm"
                value={pts[selectedIdx].x.toFixed(4)}
                step="0.01"
                on:input={(e) =>
                    actions.updatePoint(selectedIdx, "x", e.target.value)}
            />
        </div>
        <div class="mb-2">
            <label class="form-label small m-0 text-secondary">Y</label>
            <input
                type="number"
                class="form-control form-control-sm"
                value={pts[selectedIdx].y.toFixed(4)}
                step="0.01"
                on:input={(e) =>
                    actions.updatePoint(selectedIdx, "y", e.target.value)}
            />
        </div>
        <div
            class="status-badge mt-2 text-center small {winding === 'CCW'
                ? 'ok'
                : 'warn'}"
        >
            {winding === "CCW" ? "✅ CCW" : "⚠️ CW"}
        </div>
    </div>
{/if}

<style>
    .props-panel {
        width: 180px;
        z-index: 10;
        backdrop-filter: blur(4px);
    }
    .status-badge {
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
    }
    .status-badge.ok {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
    }
    .status-badge.warn {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
    }
</style>
