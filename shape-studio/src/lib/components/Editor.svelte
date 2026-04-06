<script>
    import { onMount, tick } from "svelte";
    import {
        Circle,
        Heart,
        Star,
        Undo,
        Redo,
        Upload,
        Download,
        Grid3x3,
        CheckCircle2,
        AlertTriangle,
        Move,
        X,
    } from "lucide-svelte";

    // ================= STATE =================
    let canvas;
    let ctx;
    const SCALE = 150;
    let showHighRes = false;
    let normalizeExport = true;
    let selectedIdx = -1;
    let dragIdx = -1;

    // Toast State
    let toastMsg = "";
    let toastType = "success"; // 'success' | 'error'
    let showToast = false;
    let toastTimeout;

    // Points State (Anchors are even indices)
    let pts = new Array(32)
        .fill(null)
        .map((_, i) => ({ x: 0, y: 0, anchor: i % 2 === 0 }));

    // History
    let history = [];
    let historyIdx = -1;

    // ================= LIFECYCLE =================
    onMount(() => {
        ctx = canvas.getContext("2d");
        loadPreset("circle"); // Initial load triggers draw
        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    });

    // ================= UTILS =================
    function fireToast(msg, type = "success") {
        clearTimeout(toastTimeout);
        toastMsg = msg;
        toastType = type;
        showToast = true;
        toastTimeout = setTimeout(() => (showToast = false), 2500);
    }

    function toScreen(p) {
        return {
            x: canvas.width / 2 + p.x * SCALE,
            y: canvas.height / 2 - p.y * SCALE,
        };
    }

    function toWorld(sx, sy) {
        return {
            x: (sx - canvas.width / 2) / SCALE,
            y: (canvas.height / 2 - sy) / SCALE,
        };
    }

    // ================= HISTORY =================
    function deepCopyState() {
        return pts.map((p) => ({ x: p.x, y: p.y }));
    }

    function saveState() {
        // If we undid, slice off the future
        if (historyIdx < history.length - 1) {
            history = history.slice(0, historyIdx + 1);
        }
        history.push(deepCopyState());
        historyIdx = history.length - 1;
        // Keep history reasonable
        if (history.length > 50) history.shift();
    }

    function restoreState() {
        const state = history[historyIdx];
        for (let i = 0; i < pts.length; i++) {
            pts[i].x = state[i].x;
            pts[i].y = state[i].y;
        }
        draw();
    }

    function undo() {
        if (historyIdx > 0) {
            historyIdx--;
            restoreState();
        }
    }

    function redo() {
        if (historyIdx < history.length - 1) {
            historyIdx++;
            restoreState();
        }
    }

    // ================= DRAWING =================
    function draw() {
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();

        // Grid
        ctx.strokeStyle = "#222";
        ctx.lineWidth = 1;
        for (let i = -canvas.width / 2; i < canvas.width / 2; i += SCALE / 2) {
            ctx.beginPath();
            ctx.moveTo(canvas.width / 2 + i, 0);
            ctx.lineTo(canvas.width / 2 + i, canvas.height);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, canvas.height / 2 - i);
            ctx.lineTo(canvas.width, canvas.height / 2 - i);
            ctx.stroke();
        }

        // Main 32-pt curve
        ctx.strokeStyle = "#4ade80";
        ctx.lineWidth = 2;
        ctx.beginPath();
        let s0 = toScreen(pts[0]);
        ctx.moveTo(s0.x, s0.y);
        for (let i = 0; i < 16; i++) {
            let c = toScreen(pts[2 * i + 1]);
            let e = toScreen(pts[(2 * i + 2) % 32]);
            ctx.quadraticCurveTo(c.x, c.y, e.x, e.y);
        }
        ctx.stroke();

        // High-Res Preview
        if (showHighRes) {
            const resPts = evaluateHighRes(128);
            if (resPts.length > 0) {
                ctx.save();
                ctx.strokeStyle = "rgba(148, 163, 184, 0.6)";
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 3]);
                ctx.beginPath();
                let rs0 = toScreen(resPts[0]);
                ctx.moveTo(rs0.x, rs0.y);
                for (let i = 1; i < resPts.length; i++) {
                    let rs = toScreen(resPts[i]);
                    ctx.lineTo(rs.x, rs.y);
                }
                ctx.stroke();
                ctx.setLineDash([]);

                ctx.fillStyle = "rgba(251, 191, 36, 0.7)";
                for (let i = 0; i < resPts.length; i += 4) {
                    let s = toScreen(resPts[i]);
                    ctx.beginPath();
                    ctx.arc(s.x, s.y, 2.5, 0, Math.PI * 2);
                    ctx.fill();
                }
                ctx.restore();
            }
        }

        // Control lines
        ctx.strokeStyle = "#666";
        ctx.setLineDash([5, 4]);
        ctx.lineWidth = 1;
        for (let i = 0; i < 16; i++) {
            let c = toScreen(pts[2 * i + 1]);
            let a1 = toScreen(pts[2 * i]);
            let a2 = toScreen(pts[(2 * i + 2) % 32]);
            ctx.beginPath();
            ctx.moveTo(a1.x, a1.y);
            ctx.lineTo(c.x, c.y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(a2.x, a2.y);
            ctx.lineTo(c.x, c.y);
            ctx.stroke();
        }
        ctx.setLineDash([]);

        // Points
        pts.forEach((p, i) => {
            let s = toScreen(p);
            let isSelected = i === selectedIdx;
            let radius = p.anchor ? 6 : 4;

            ctx.beginPath();
            ctx.arc(s.x, s.y, radius, 0, Math.PI * 2);
            ctx.fillStyle = p.anchor ? "#3b82f6" : "#ef4444";
            if (isSelected) ctx.fillStyle = "#ffffff";
            ctx.fill();

            ctx.strokeStyle = isSelected ? "#2563eb" : "#fff";
            ctx.lineWidth = isSelected ? 2 : 1;
            ctx.stroke();
        });

        // Validation UI
        const { winding } = validateShape();
        const statusEl = document.getElementById("validationStatus");
        if (statusEl) {
            statusEl.textContent = winding === "CCW" ? "✅ CCW" : "⚠️ CW";
            statusEl.className = `status-badge ${winding === "CCW" ? "ok" : "warn"}`;
        }

        ctx.restore();
    }

    // ================= INTERACTION =================
    function handlePointerDown(e) {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        dragIdx = -1;

        for (let i = 0; i < pts.length; i++) {
            let s = toScreen(pts[i]);
            if (Math.hypot(mx - s.x, my - s.y) < 12) {
                dragIdx = i;
                selectedIdx = i;
                break;
            }
        }

        if (dragIdx === -1) selectedIdx = -1;

        if (dragIdx !== -1) {
            saveState();
            canvas.setPointerCapture(e.pointerId);
        }
        draw();
    }

    function handlePointerMove(e) {
        if (dragIdx === -1) return;
        const rect = canvas.getBoundingClientRect();
        const w = toWorld(e.clientX - rect.left, e.clientY - rect.top);

        // Direct mutation is fine here as we call draw() explicitly
        pts[dragIdx].x = w.x;
        pts[dragIdx].y = w.y;

        draw();
    }

    function handlePointerUp() {
        dragIdx = -1;
    }

    function updatePointFromInput(idx, axis, value) {
        let val = parseFloat(value);
        if (isNaN(val)) return;
        if (pts[idx]) {
            pts[idx][axis] = val;
            draw();
        }
    }

    // ================= LOGIC =================
    function evaluateHighRes(totalPoints) {
        let rawPoints = [];
        const samplesPerSegment = 20;

        for (let i = 0; i < 16; i++) {
            let p0 = pts[2 * i],
                p1 = pts[2 * i + 1],
                p2 = pts[(2 * (i + 1)) % 32];
            for (let j = 0; j < samplesPerSegment; j++) {
                let t = j / samplesPerSegment;
                let x =
                    (1 - t) ** 2 * p0.x +
                    2 * (1 - t) * t * p1.x +
                    t ** 2 * p2.x;
                let y =
                    (1 - t) ** 2 * p0.y +
                    2 * (1 - t) * t * p1.y +
                    t ** 2 * p2.y;
                rawPoints.push({ x, y });
            }
        }
        rawPoints.push(rawPoints[0]);

        let lengths = [0];
        let totalLength = 0;
        for (let i = 1; i < rawPoints.length; i++) {
            let d = Math.hypot(
                rawPoints[i].x - rawPoints[i - 1].x,
                rawPoints[i].y - rawPoints[i - 1].y,
            );
            totalLength += d;
            lengths.push(totalLength);
        }

        let resampled = [];
        for (let i = 0; i < totalPoints; i++) {
            let targetDist = (i / totalPoints) * totalLength;
            let idx = lengths.findIndex((l) => l >= targetDist);
            if (idx === -1) idx = lengths.length - 1;
            if (idx === 0) {
                resampled.push(rawPoints[0]);
                continue;
            }
            let l1 = lengths[idx - 1];
            let l2 = lengths[idx];
            let ratio = (targetDist - l1) / (l2 - l1);
            let p1 = rawPoints[idx - 1];
            let p2 = rawPoints[idx];
            resampled.push({
                x: p1.x + (p2.x - p1.x) * ratio,
                y: p1.y + (p2.y - p1.y) * ratio,
            });
        }
        return resampled;
    }

    function validateShape() {
        let x = pts.map((p) => p.x),
            y = pts.map((p) => p.y);
        let area =
            0.5 *
            (x
                .slice(0, -1)
                .reduce((s, v, i) => s + v * y[i + 1] - x[i + 1] * y[i], 0) +
                x[x.length - 1] * y[0] -
                x[0] * y[y.length - 1]);
        return { winding: area > 0 ? "CCW" : "CW" };
    }

    function loadPreset(type) {
        saveState();
        for (let i = 0; i < 32; i++) {
            let angle = (i * Math.PI) / 16;
            if (type === "circle") {
                let r = 0.85;
                pts[i].x = r * Math.cos(angle);
                pts[i].y = r * Math.sin(angle);
            } else if (type === "heart") {
                let hx = 16 * Math.pow(Math.sin(angle), 3);
                let hy =
                    13 * Math.cos(angle) -
                    5 * Math.cos(2 * angle) -
                    2 * Math.cos(3 * angle) -
                    Math.cos(4 * angle);
                let maxExt = 16.5;
                if (i % 2 === 0) {
                    pts[i].x = hx / maxExt;
                    pts[i].y = hy / maxExt;
                } else {
                    let prev = i - 1,
                        next = (i + 1) % 32;
                    let aPrevX =
                        (16 * Math.pow(Math.sin((prev * Math.PI) / 16), 3)) /
                        maxExt;
                    let aPrevY =
                        (13 * Math.cos((prev * Math.PI) / 16) -
                            5 * Math.cos((2 * prev * Math.PI) / 16) -
                            2 * Math.cos((3 * prev * Math.PI) / 16) -
                            Math.cos((4 * prev * Math.PI) / 16)) /
                        maxExt;
                    let aNextX =
                        (16 * Math.pow(Math.sin((next * Math.PI) / 16), 3)) /
                        maxExt;
                    let aNextY =
                        (13 * Math.cos((next * Math.PI) / 16) -
                            5 * Math.cos((2 * next * Math.PI) / 16) -
                            2 * Math.cos((3 * next * Math.PI) / 16) -
                            Math.cos((4 * next * Math.PI) / 16)) /
                        maxExt;
                    pts[i].x = (aPrevX + aNextX) * 0.5;
                    pts[i].y = (aPrevY + aNextY) * 0.5;
                }
            } else if (type === "star") {
                let anchorIdx = Math.floor(i / 2),
                    isOuter = anchorIdx % 2 === 0;
                let r =
                    i % 2 === 0
                        ? isOuter
                            ? 0.9
                            : 0.45
                        : isOuter
                          ? 0.75
                          : 0.55;
                pts[i].x = r * Math.cos(angle);
                pts[i].y = r * Math.sin(angle);
            }
        }
        selectedIdx = -1;
        draw();
    }

    function downloadJSON() {
        let data = pts.map((p) => [
            parseFloat(p.x.toFixed(4)),
            parseFloat(p.y.toFixed(4)),
        ]);
        let meta = {
            version: "1.0",
            type: "quad_bezier",
            segments: 16,
            normalized: normalizeExport,
            winding: validateShape().winding,
            timestamp: Date.now(),
        };

        if (normalizeExport) {
            let maxVal = Math.max(...data.flat().map(Math.abs)) || 1;
            data = data.map(([x, y]) => [x / maxVal, y / maxVal]);
        }

        const payload = { meta, points: data };
        const blob = new Blob([JSON.stringify(payload, null, 2)], {
            type: "application/json",
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `shape_32pt_v${Date.now().toString().slice(-4)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        fireToast("✅ Exported successfully");
    }

    function handleImport(file) {
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const json = JSON.parse(event.target.result);
                let points = json.points || json;
                if (!Array.isArray(points) || points.length !== 32)
                    throw new Error("Expected exactly 32 points");
                points.forEach((p, i) => {
                    if (!Array.isArray(p) || p.length !== 2)
                        throw new Error(`Point ${i} must be [x, y]`);
                    pts[i].x = parseFloat(p[0]) || 0;
                    pts[i].y = parseFloat(p[1]) || 0;
                });
                history = [];
                historyIdx = -1;
                saveState();
                selectedIdx = -1;
                draw();
                fireToast("✅ Shape imported successfully");
            } catch (err) {
                fireToast("❌ Import failed: " + err.message, "error");
            }
        };
        reader.readAsText(file);
    }

    function handleKeyDown(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === "z") {
            e.preventDefault();
            undo();
        }
        if ((e.ctrlKey || e.metaKey) && e.key === "y") {
            e.preventDefault();
            redo();
        }
        if (e.key === "Escape") {
            selectedIdx = -1;
            draw();
        }
    }

    // Reactive triggers
    $: history, historyIdx, selectedIdx, pts;
</script>

<div class="editor-container">
    <!-- Toolbar -->
    <div
        class="toolbar p-2 border-bottom border-secondary bg-dark-subtle d-flex flex-wrap align-items-center gap-2"
    >
        <div class="btn-group btn-group-sm me-2">
            <button
                class="btn btn-outline-secondary"
                title="Circle"
                on:click={() => loadPreset("circle")}
            >
                <Circle size={16} />
            </button>
            <button
                class="btn btn-outline-secondary"
                title="Heart"
                on:click={() => loadPreset("heart")}
            >
                <Heart size={16} />
            </button>
            <button
                class="btn btn-outline-secondary"
                title="Star"
                on:click={() => loadPreset("star")}
            >
                <Star size={16} />
            </button>
        </div>

        <div class="vr bg-secondary mx-1"></div>

        <div class="btn-group btn-group-sm me-2">
            <button
                class="btn btn-outline-secondary"
                disabled={historyIdx <= 0}
                title="Undo"
                on:click={undo}
            >
                <Undo size={16} />
            </button>
            <button
                class="btn btn-outline-secondary"
                disabled={historyIdx >= history.length - 1}
                title="Redo"
                on:click={redo}
            >
                <Redo size={16} />
            </button>
        </div>

        <div class="vr bg-secondary mx-1"></div>

        <div
            class="form-check form-switch d-inline-flex align-items-center me-2 mb-0"
        >
            <input
                class="form-check-input me-1"
                type="checkbox"
                id="highRes"
                bind:checked={showHighRes}
            />
            <label class="form-check-label small" for="highRes"
                ><Grid3x3 size={12} class="d-inline me-1" />High-Res</label
            >
        </div>

        <div
            class="form-check form-switch d-inline-flex align-items-center mb-0"
        >
            <input
                class="form-check-input me-1"
                type="checkbox"
                id="normalize"
                bind:checked={normalizeExport}
            />
            <label class="form-check-label small" for="normalize"
                >Normalize</label
            >
        </div>

        <div class="flex-grow-1"></div>

        <div class="btn-group btn-group-sm">
            <button class="btn btn-outline-secondary" title="Import JSON">
                <Upload size={16} />
                <input
                    type="file"
                    accept=".json"
                    class="d-none"
                    on:change={(e) => handleImport(e.target.files[0])}
                />
            </button>
            <button
                class="btn btn-primary"
                title="Export JSON"
                on:click={downloadJSON}
            >
                <Download size={16} class="me-1" /> Export
            </button>
        </div>
    </div>

    <!-- Canvas Area -->
    <div class="canvas-wrapper position-relative bg-dark">
        <canvas
            bind:this={canvas}
            width={800}
            height={800}
            class="border border-secondary"
            on:pointerdown={handlePointerDown}
            on:pointermove={handlePointerMove}
            on:pointerup={handlePointerUp}
        ></canvas>

        <!-- Properties Panel -->
        {#if selectedIdx !== -1}
            <div
                class="props-panel position-absolute top-0 end-0 m-2 p-2 bg-dark border border-secondary rounded shadow"
            >
                <div
                    class="d-flex justify-content-between align-items-center mb-2"
                >
                    <strong class="small">Point #{selectedIdx}</strong>
                    <button
                        class="btn-close btn-close-white"
                        style="font-size: 0.5rem;"
                        on:click={() => (selectedIdx = -1)}
                    ></button>
                </div>
                <div class="mb-1">
                    <!-- svelte-ignore a11y-label-has-associated-control -->
                    <label class="form-label small m-0 text-secondary">X</label>
                    <input
                        type="number"
                        class="form-control form-control-sm bg-dark text-white border-secondary"
                        value={pts[selectedIdx].x.toFixed(4)}
                        step="0.01"
                        on:input={(e) =>
                            updatePointFromInput(
                                selectedIdx,
                                "x",
                                e.target.value,
                            )}
                    />
                </div>
                <div>
                    <!-- svelte-ignore a11y-label-has-associated-control -->
                    <label class="form-label small m-0 text-secondary">Y</label>
                    <input
                        type="number"
                        class="form-control form-control-sm bg-dark text-white border-secondary"
                        value={pts[selectedIdx].y.toFixed(4)}
                        step="0.01"
                        on:input={(e) =>
                            updatePointFromInput(
                                selectedIdx,
                                "y",
                                e.target.value,
                            )}
                    />
                </div>
                <div
                    id="validationStatus"
                    class="status-badge mt-2 text-center small"
                >
                    ⏳ Ready
                </div>
            </div>
        {/if}
    </div>

    <!-- Toast -->
    {#if showToast}
        <div class="toast-container">
            <div
                class="toast {toastType === 'error'
                    ? 'bg-danger'
                    : 'bg-success'} text-white border-0 show"
                role="alert"
            >
                <div class="d-flex align-items-center p-2">
                    {#if toastType === "error"}
                        <AlertTriangle size={16} class="me-2" />
                    {:else}
                        <CheckCircle2 size={16} class="me-2" />
                    {/if}
                    <span class="small fw-bold">{toastMsg}</span>
                </div>
            </div>
        </div>
    {/if}
</div>

<style>
    :global(body) {
        overflow: hidden;
    }

    .editor-container {
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    .canvas-wrapper {
        flex-grow: 1;
        display: flex;
        justify-content: center;
        align-items: center;
        overflow: hidden;
        position: relative;
    }

    canvas {
        cursor: crosshair;
        background: #111;
    }

    .props-panel {
        width: 180px;
        z-index: 10;
        backdrop-filter: blur(4px);
    }

    .toast-container {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1050;
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

    /* Override Bootstrap inputs for dark theme */
    :global(input.form-control) {
        background-color: #1a1a1a !important;
        color: #eee !important;
    }
    :global(input.form-control:focus) {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 0.15rem rgba(37, 99, 235, 0.25) !important;
    }
</style>
