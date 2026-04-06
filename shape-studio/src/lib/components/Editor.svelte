<script>
    import { onMount, onDestroy } from "svelte";
    import { actions } from "../stores/editorStore.js";

    // Import internal components
    import EditorToolbar from "./EditorToolbar.svelte";
    import EditorCanvas from "./EditorCanvas.svelte";
    import EditorPanel from "./EditorPanel.svelte";
    import EditorToast from "./EditorToast.svelte";
    function handleKeyDown(e) {
        const mod = e.ctrlKey || e.metaKey;
        if (mod && e.key === "z") {
            e.preventDefault();
            actions.undo();
        }
        if (mod && e.key === "y") {
            e.preventDefault();
            actions.redo();
        }
        if (e.key === "Escape") actions.selectPoint(-1);
    }

    onMount(() => window.addEventListener("keydown", handleKeyDown));
    onDestroy(() => window.removeEventListener("keydown", handleKeyDown));
</script>

<div class="editor-container">
    <EditorToolbar />
    <!-- flex: 1 ensures this wrapper takes all remaining height -->
    <div class="canvas-wrapper position-relative">
        <EditorCanvas />
        <EditorPanel />
        <EditorToast />
    </div>
</div>

<style>
    :global(body) {
        overflow: hidden;
        margin: 0;
    }
    .editor-container {
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .canvas-wrapper {
        flex: 1;
        display: flex;
        background: #0a0a0a;
        overflow: hidden; /* Prevents scrollbars */
    }
    :global(input.form-control) {
        background-color: #1a1a1a !important;
        color: #eee !important;
    }
    :global(input.form-control:focus) {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 0.15rem rgba(37, 99, 235, 0.25) !important;
    }
</style>
