<script>
    import {
        Circle,
        Heart,
        Star,
        Undo,
        Redo,
        Upload,
        Download,
        Grid3x3,
        FileJson,
        Eye,
        X,
    } from "lucide-svelte";
    import { shapeState, actions } from "../stores/editorStore.js";

    $: ({ showHighRes, normalizeExport, historyIdx, history, reference } =
        $shapeState);
    $: refCount = reference?.points?.length || 0;
    $: hasRef = refCount > 0;
</script>

<div
    class="toolbar p-2 border-bottom border-secondary bg-dark-subtle d-flex flex-wrap align-items-center gap-2"
>
    <!-- Presets -->
    <div class="btn-group btn-group-sm me-2">
        <button
            class="btn btn-outline-secondary"
            title="Circle"
            on:click={() => actions.loadPreset("circle")}
            ><Circle size={16} /></button
        >
        <button
            class="btn btn-outline-secondary"
            title="Heart"
            on:click={() => actions.loadPreset("heart")}
            ><Heart size={16} /></button
        >
        <button
            class="btn btn-outline-secondary"
            title="Star"
            on:click={() => actions.loadPreset("star")}
            ><Star size={16} /></button
        >
    </div>

    <div class="vr bg-secondary mx-1"></div>

    <!-- History -->
    <div class="btn-group btn-group-sm me-2">
        <button
            class="btn btn-outline-secondary"
            disabled={historyIdx <= 0}
            title="Undo"
            on:click={actions.undo}><Undo size={16} /></button
        >
        <button
            class="btn btn-outline-secondary"
            disabled={historyIdx >= history.length - 1}
            title="Redo"
            on:click={actions.redo}><Redo size={16} /></button
        >
    </div>

    <div class="vr bg-secondary mx-1"></div>

    <!-- Toggles -->
    <label
        class="form-check form-switch d-inline-flex align-items-center me-2 mb-0"
    >
        <input
            class="form-check-input me-1"
            type="checkbox"
            bind:checked={showHighRes}
            on:change={actions.toggleHighRes}
        />
        <span class="form-check-label small"
            ><Grid3x3 size={12} class="d-inline me-1" />High-Res</span
        >
    </label>

    <!-- ✅ Reference Controls -->
    <div class="vr bg-secondary mx-1"></div>
    <label
        class="btn btn-outline-info btn-sm d-inline-flex align-items-center"
        title="Import Reference JSON"
    >
        <FileJson size={14} class="me-1" /> Ref
        <input
            type="file"
            accept=".json"
            class="d-none"
            on:change={(e) => actions.loadReference(e.target.files[0])}
        />
    </label>

    {#if hasRef}
        <button
            class="btn btn-outline-info btn-sm d-inline-flex align-items-center"
            title="Toggle Reference"
            on:click={actions.toggleReference}
        >
            <Eye size={14} class={reference.visible ? "" : "opacity-50"} />
        </button>
        <span
            class="badge bg-info bg-opacity-25 text-info border border-info small px-1"
            >{refCount}pts</span
        >
        <button
            class="btn btn-outline-danger btn-sm px-1 py-0"
            title="Clear Reference"
            on:click={actions.clearReference}
            style="line-height:1;"
        >
            <X size={12} />
        </button>
    {/if}

    <div class="vr bg-secondary mx-1"></div>

    <label class="form-check form-switch d-inline-flex align-items-center mb-0">
        <input
            class="form-check-input me-1"
            type="checkbox"
            bind:checked={normalizeExport}
            on:change={actions.toggleNormalize}
        />
        <span class="form-check-label small">Normalize Export</span>
    </label>

    <div class="flex-grow-1"></div>

    <!-- I/O -->
    <div class="btn-group btn-group-sm">
        <label class="btn btn-outline-secondary" title="Import Shape JSON">
            <Upload size={16} />
            <input
                type="file"
                accept=".json"
                class="d-none"
                on:change={(e) => actions.handleImport(e.target.files[0])}
            />
        </label>
        <button
            class="btn btn-primary"
            title="Export Shape JSON"
            on:click={actions.downloadJSON}
        >
            <Download size={16} class="me-1" /> Export
        </button>
    </div>
</div>
