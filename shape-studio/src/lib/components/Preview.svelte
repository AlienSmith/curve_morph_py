<script>
    import { Row, Col, Button, Input, Label } from "sveltestrap";
    import { Zap, Download, FileCode } from "lucide-svelte";
    import {
        morphResult,
        morphBinary,
        status,
        renderDpi,
    } from "../stores/previewStore.js";

    let fileA, fileB;

    async function runFFTMorph() {
        if (!fileA || !fileB) return alert("Select two JSON files!");
        status.set("⚡ Processing morph...");

        const fd = new FormData();
        fd.append("fileA", fileA);
        fd.append("fileB", fileB);
        fd.append("dpi", $renderDpi);
        fd.append("include_morph", "true"); // Explicitly request .morph file

        try {
            const resp = await fetch("/api/generate-morph", {
                method: "POST",
                body: fd,
            });

            // Parse multipart response to get both files
            const formData = await resp.formData();

            // Get GIF preview
            const gifBlob = formData.get("gif");
            morphResult.set(URL.createObjectURL(gifBlob));

            // Get .morph binary
            const morphBlob = formData.get("morph");
            morphBinary.set(URL.createObjectURL(morphBlob));

            status.set("✅ Morph generated!");
        } catch (e) {
            console.error("Morph error:", e);
            status.set("❌ Processing failed");
        }
    }

    // Clean up blob URLs when component unmounts to avoid memory leaks
    function cleanupBlobs() {
        if ($morphResult) URL.revokeObjectURL($morphResult);
        if ($morphBinary) URL.revokeObjectURL($morphBinary);
    }
    window.addEventListener("beforeunload", cleanupBlobs);
</script>

<Row class="g-0 h-100">
    <Col md="3" class="bg-dark p-4 border-end border-secondary">
        <h6 class="text-secondary mb-4">MORPH SETTINGS</h6>

        <div class="mb-3">
            <Label size="sm">START SHAPE (.JSON)</Label>
            <Input
                type="file"
                accept=".json"
                on:change={(e) => (fileA = e.target.files[0])}
            />
        </div>

        <div class="mb-3">
            <Label size="sm">END SHAPE (.JSON)</Label>
            <Input
                type="file"
                accept=".json"
                on:change={(e) => (fileB = e.target.files[0])}
            />
        </div>

        <div class="mb-4">
            <Label size="sm">DPI: {$renderDpi}</Label>
            <Input type="range" min="72" max="200" bind:value={$renderDpi} />
        </div>

        <Button color="primary" class="w-100 py-2" on:click={runFFTMorph}>
            <Zap size={16} class="me-2" /> Generate Morph
        </Button>

        <div class="mt-3 text-center small text-secondary">{$status}</div>
    </Col>

    <Col
        md="9"
        class="d-flex align-items-center justify-content-center bg-black position-relative"
    >
        {#if $morphResult}
            <div class="text-center">
                <img
                    src={$morphResult}
                    alt="Morph preview"
                    class="img-fluid rounded shadow-lg border border-secondary"
                />
                <div class="mt-3 gap-2 d-flex justify-content-center">
                    <Button
                        color="success"
                        size="sm"
                        href={$morphResult}
                        download="morph.gif"
                    >
                        <Download size={14} class="me-1" /> Download .gif
                    </Button>
                    <!-- New .morph download button -->
                    <Button
                        color="info"
                        size="sm"
                        href={$morphBinary}
                        download="animation.morph"
                        disabled={!$morphBinary}
                    >
                        <Download size={14} class="me-1" /> Download .morph
                    </Button>
                </div>
            </div>
        {:else}
            <div
                class="text-muted border border-secondary border-dashed p-5 rounded"
            >
                <FileCode size={48} class="mb-3 opacity-25" />
                <p>Awaiting shape data for processing...</p>
            </div>
        {/if}
    </Col>
</Row>
