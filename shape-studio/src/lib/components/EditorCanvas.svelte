<script>
    import { onMount, onDestroy } from "svelte";
    import { shapeState, actions } from "../stores/editorStore.js";
    import { evaluateHighRes } from "../utils/geometry.js";

    let canvas;
    let ctx;
    let dragIdx = -1;
    let isPanning = false;
    let panStart = { x: 0, y: 0, panX: 0, panY: 0 };

    const BASE_SCALE = 150;
    let viewport = { x: 0, y: 0, zoom: 1 };
    let W = 800,
        H = 800;
    let resizeObserver;

    $: ({ pts, showHighRes, selectedIdx, reference } = $shapeState);
    $: refPts = reference?.points || null;
    $: showRef = reference?.visible || false;

    // ✅ TRIGGER: Redraw when pts, viewport, or reference changes
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

    // --- TRANSFORM HELPERS ---
    function getEffectiveScale() {
        return BASE_SCALE * viewport.zoom;
    }

    function toScreen(p) {
        const s = getEffectiveScale();
        return {
            x: W / 2 + viewport.x + p.x * s,
            y: H / 2 + viewport.y - p.y * s,
        };
    }

    function toWorld(sx, sy) {
        const s = getEffectiveScale();
        return {
            x: (sx - W / 2 - viewport.x) / s,
            y: (H / 2 + viewport.y - sy) / s,
        };
    }

    // --- DRAWING ---
    function draw() {
        if (!ctx || !pts) return;
        ctx.clearRect(0, 0, W, H);
        ctx.save();

        // 1. Adaptive Grid
        const tl = toWorld(0, 0);
        const br = toWorld(W, H);
        const gridStep = 0.5; // World units
        const minX = Math.floor(Math.min(tl.x, br.x) / gridStep) * gridStep;
        const maxX = Math.ceil(Math.max(tl.x, br.x) / gridStep) * gridStep;
        const minY = Math.floor(Math.min(tl.y, br.y) / gridStep) * gridStep;
        const maxY = Math.ceil(Math.max(tl.y, br.y) / gridStep) * gridStep;

        ctx.strokeStyle = "#222";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let x = minX; x <= maxX; x += gridStep) {
            const sx = toScreen({ x, y: 0 }).x;
            ctx.moveTo(sx, 0);
            ctx.lineTo(sx, H);
        }
        for (let y = minY; y <= maxY; y += gridStep) {
            const sy = toScreen({ x: 0, y }).y;
            ctx.moveTo(0, sy);
            ctx.lineTo(W, sy);
        }
        ctx.stroke();

        // 2. Reference Layer
        if (showRef && refPts && refPts.length === 32) {
            ctx.save();
            ctx.strokeStyle = "rgba(255, 255, 255, 1.0)";
            ctx.lineWidth = 1.5;
            ctx.setLineDash([6, 4]);
            ctx.beginPath();
            const s0 = toScreen(refPts[0]);
            ctx.moveTo(s0.x, s0.y);
            for (let i = 0; i < 16; i++) {
                const c = toScreen(refPts[2 * i + 1]);
                const e = toScreen(refPts[(2 * i + 2) % 32]);
                ctx.quadraticCurveTo(c.x, c.y, e.x, e.y);
            }
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();
        }

        // 3. Main Curve
        ctx.strokeStyle = "#4ade80";
        ctx.lineWidth = 2;
        ctx.beginPath();
        let s0 = toScreen(pts[0]);
        ctx.moveTo(s0.x, s0.y);
        for (let i = 0; i < 16; i++) {
            const c = toScreen(pts[2 * i + 1]);
            const e = toScreen(pts[(2 * i + 2) % 32]);
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
            let rs0 = toScreen(res[0]);
            ctx.moveTo(rs0.x, rs0.y);
            for (let i = 1; i < res.length; i++) {
                const r = toScreen(res[i]);
                ctx.lineTo(r.x, r.y);
            }
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = "rgba(251,191,36,0.7)";
            for (let i = 0; i < res.length; i += 1) {
                const r = toScreen(res[i]);
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
            const c = toScreen(pts[2 * i + 1]);
            const a1 = toScreen(pts[2 * i]);
            const a2 = toScreen(pts[(2 * i + 2) % 32]);
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
            const s = toScreen(p);
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

    // --- INTERACTIONS ---
    function onWheel(e) {
        e.preventDefault();
        const { offsetX: mx, offsetY: my } = e;
        const worldBefore = toWorld(mx, my);

        const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        viewport.zoom = Math.min(5, Math.max(0.1, viewport.zoom * zoomFactor));

        // Adjust pan to keep world point under cursor stable
        const screenAfter = toScreen(worldBefore);
        viewport.x -= screenAfter.x - mx;
        viewport.y -= screenAfter.y - my;

        draw();
    }

    function onDown(e) {
        if (!pts) return;
        const { offsetX: mx, offsetY: my } = e;

        // Hit test
        let hitIdx = -1;
        for (let i = 0; i < pts.length; i++) {
            const s = toScreen(pts[i]);
            if (Math.hypot(mx - s.x, my - s.y) < 16) {
                hitIdx = i;
                break;
            }
        }

        if (hitIdx !== -1 && e.button === 0) {
            dragIdx = hitIdx;
            actions.selectPoint(dragIdx);
            canvas.setPointerCapture(e.pointerId);
        } else if (e.button === 2 || (e.button === 0 && hitIdx === -1)) {
            isPanning = true;
            panStart = {
                x: e.clientX,
                y: e.clientY,
                panX: viewport.x,
                panY: viewport.y,
            };
            canvas.setPointerCapture(e.pointerId);
            canvas.style.cursor = "grabbing";
        }
    }

    function onMove(e) {
        if (dragIdx !== -1) {
            const { offsetX: mx, offsetY: my } = e;
            const w = toWorld(mx, my);
            pts[dragIdx].x = w.x;
            pts[dragIdx].y = w.y;
            draw();
        } else if (isPanning) {
            viewport.x = panStart.panX + (e.clientX - panStart.x);
            viewport.y = panStart.panY + (e.clientY - panStart.y);
            draw();
        }
    }

    function onUp() {
        if (dragIdx !== -1) {
            actions.commitDrag(dragIdx, pts[dragIdx].x, pts[dragIdx].y);
            dragIdx = -1;
        }
        if (isPanning) {
            isPanning = false;
            canvas.style.cursor = "crosshair";
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
    on:pointerleave={onUp}
    on:wheel={onWheel}
    on:contextmenu|preventDefault
></canvas>
