<script>
    import { CheckCircle2, AlertTriangle } from "lucide-svelte";
    import { shapeState } from "../stores/editorStore.js";

    $: ({ toast } = $shapeState);
</script>

{#if toast.visible}
    <div class="toast-container">
        <div
            class="toast {toast.type === 'error'
                ? 'bg-danger'
                : 'bg-success'} text-white border-0 show"
            role="alert"
        >
            <div class="d-flex align-items-center p-2">
                {#if toast.type === "error"}
                    <AlertTriangle size={16} class="me-2" />
                {:else}
                    <CheckCircle2 size={16} class="me-2" />
                {/if}
                <span class="small fw-bold">{toast.msg}</span>
            </div>
        </div>
    </div>
{/if}

<style>
    .toast-container {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1050;
    }
</style>
