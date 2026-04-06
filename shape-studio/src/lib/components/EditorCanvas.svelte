<script>
    import { onMount, onDestroy } from "svelte";
    import { shapeState } from "../stores/editorStore.js";
    import { toScreen, evaluateHighRes, toWorld } from "../utils/geometry.js";
    import { actions } from "../stores/editorStore.js";

    let canvas;
    let ctx;
    let dragIdx = -1;
    const SCALE = 150;
    let W = 800,
        H = 800;
    let resizeObserver;

    $: ({ pts, showHighRes, selectedIdx, reference } = $shapeState);
    $: refPts = reference?.points || null;
    $: showRef = reference?.visible || false;

    // ✅ TRIGGER: Redraw when pts OR reference changes
    $: if (pts && ctx) draw();

    onMount(() => {
        if (!canvas) return;
        ctx = canvas.getContext("2d");
        resizeCanvas();
        draw();

        resizeObserver = new ResizeObserver(() => {
            resizeCanvas();
            draw();
        });
        resizeObserver.observe(canvas.parentElement);
    });

    onDestroy(() => resizeObserver?.disconnect());

    function resizeCanvas() {
        const rect = canvas.getBoundingClientRect();
        W = rect.width;
        H = rect.height;
        canvas.width = W;
        canvas.height = H;
    }

    function draw() {
        if (!ctx || !pts) return;
        ctx.clearRect(0, 0, W, H);
        ctx.save();

        // 1. Grid
        ctx.strokeStyle = "#222";
        ctx.lineWidth = 1;
        for (let i = -W / 2; i < W / 2; i += SCALE / 2) {
            ctx.beginPath();
            ctx.moveTo(W / 2 + i, 0);
            ctx.lineTo(W / 2 + i, H);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, H / 2 - i);
            ctx.lineTo(W, H / 2 - i);
            ctx.stroke();
        }

        // ✅ Reference Layer (Static, Non-Interactive)
        if (showRef && refPts && refPts.length > 1) {
            ctx.save();
            ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
            ctx.lineWidth = 1.5;
            ctx.setLineDash([6, 4]);
            ctx.beginPath();
            const s0 = toScreen(refPts[0], W, H, SCALE);
            ctx.moveTo(s0.x, s0.y);
            for (let i = 1; i < refPts.length; i++) {
                const s = toScreen(refPts[i], W, H, SCALE);
                ctx.lineTo(s.x, s.y);
            }
            // Auto-close loop if start/end match (common in exported shapes)
            const first = toScreen(refPts[0], W, H, SCALE);
            const last = toScreen(refPts[refPts.length - 1], W, H, SCALE);
            if (Math.hypot(first.x - last.x, first.y - last.y) < 2)
                ctx.closePath();
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();
        }

        // 3. Main Curve
        ctx.strokeStyle = "#4ade80";
        ctx.lineWidth = 2;
        ctx.beginPath();
        let s0 = toScreen(pts[0], W, H, SCALE);
        ctx.moveTo(s0.x, s0.y);
        for (let i = 0; i < 16; i++) {
            const c = toScreen(pts[2 * i + 1], W, H, SCALE);
            const e = toScreen(pts[(2 * i + 2) % 32], W, H, SCALE);
            ctx.quadraticCurveTo(c.x, c.y, e.x, e.y);
        }
        ctx.stroke();

        // 4. High-Res
        if (showHighRes) {
            const res = evaluateHighRes(pts, 128);
            ctx.save();
            ctx.strokeStyle = "rgba(148,163,184,0.6)";
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            let rs0 = toScreen(res[0], W, H, SCALE);
            ctx.moveTo(rs0.x, rs0.y);
            for (let i = 1; i < res.length; i++) {
                const r = toScreen(res[i], W, H, SCALE);
                ctx.lineTo(r.x, r.y);
            }
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = "rgba(251,191,36,0.7)";
            for (let i = 0; i < res.length; i += 4) {
                const r = toScreen(res[i], W, H, SCALE);
                ctx.beginPath();
                ctx.arc(r.x, r.y, 2.5, 0, Math.PI * 2);
                ctx.fill();
            }
            ctx.restore();
        }

        // 5. Control Lines
        ctx.strokeStyle = "#666";
        ctx.setLineDash([5, 4]);
        ctx.lineWidth = 1;
        for (let i = 0; i < 16; i++) {
            const c = toScreen(pts[2 * i + 1], W, H, SCALE);
            const a1 = toScreen(pts[2 * i], W, H, SCALE);
            const a2 = toScreen(pts[(2 * i + 2) % 32], W, H, SCALE);
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

        // 6. Points
        pts.forEach((p, i) => {
            const s = toScreen(p, W, H, SCALE);
            const sel = i === selectedIdx;
            ctx.beginPath();
            ctx.arc(s.x, s.y, p.anchor ? 6 : 4, 0, Math.PI * 2);
            ctx.fillStyle = sel ? "#fff" : p.anchor ? "#3b82f6" : "#ef4444";
            ctx.fill();
            ctx.strokeStyle = sel ? "#2563eb" : "#fff";
            ctx.lineWidth = sel ? 2 : 1;
            ctx.stroke();
        });
        ctx.restore();
    }

    function getCanvasCoords(e) {
        const rect = canvas.getBoundingClientRect();
        return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    function onDown(e) {
        if (!pts) return;
        const { x, y } = getCanvasCoords(e);
        dragIdx = -1;
        for (let i = 0; i < pts.length; i++) {
            const s = toScreen(pts[i], W, H, SCALE);
            if (Math.hypot(x - s.x, y - s.y) < 16) {
                dragIdx = i;
                break;
            }
        }
        actions.selectPoint(dragIdx);
        if (dragIdx !== -1) canvas.setPointerCapture(e.pointerId);
    }

    function onMove(e) {
        if (dragIdx === -1) return;
        const { x, y } = getCanvasCoords(e);
        const w = toWorld(x, y, W, H, SCALE);
        pts[dragIdx].x = w.x;
        pts[dragIdx].y = w.y;
        draw();
    }

    function onUp() {
        if (dragIdx !== -1) {
            actions.commitDrag(dragIdx, pts[dragIdx].x, pts[dragIdx].y);
            dragIdx = -1;
        }
    }
</script>

<canvas
    bind:this={canvas}
    class="w-100 h-100 d-block"
    style="background: #111; cursor: crosshair; touch-action: none;"
    on:pointerdown={onDown}
    on:pointermove={onMove}
    on:pointerup={onUp}
></canvas>
