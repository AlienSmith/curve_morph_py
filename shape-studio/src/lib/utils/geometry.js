export function toScreen(p, w, h, SCALE) {
    return { x: w / 2 + p.x * SCALE, y: h / 2 - p.y * SCALE };
}

export function toWorld(sx, sy, w, h, SCALE) {
    return { x: (sx - w / 2) / SCALE, y: (h / 2 - sy) / SCALE };
}

export function deepCopyState(pts) {
    return pts.map(p => ({ x: p.x, y: p.y }));
}

export function validateShape(pts) {
    const x = pts.map(p => p.x), y = pts.map(p => p.y);
    const area = 0.5 * (
        x.slice(0, -1).reduce((s, v, i) => s + v * y[i + 1] - x[i + 1] * y[i], 0) +
        x[x.length - 1] * y[0] - x[0] * y[y.length - 1]
    );
    return { winding: area > 0 ? "CCW" : "CW" };
}

export function evaluateHighRes(pts, totalPoints = 128) {
    const samples = 20;
    let raw = [];
    for (let i = 0; i < 16; i++) {
        const p0 = pts[2 * i], p1 = pts[2 * i + 1], p2 = pts[(2 * (i + 1)) % 32];
        for (let j = 0; j < samples; j++) {
            const t = j / samples;
            raw.push({
                x: (1 - t) ** 2 * p0.x + 2 * (1 - t) * t * p1.x + t ** 2 * p2.x,
                y: (1 - t) ** 2 * p0.y + 2 * (1 - t) * t * p1.y + t ** 2 * p2.y
            });
        }
    }
    raw.push(raw[0]);

    let lengths = [0], totalLen = 0;
    for (let i = 1; i < raw.length; i++) {
        const d = Math.hypot(raw[i].x - raw[i - 1].x, raw[i].y - raw[i - 1].y);
        totalLen += d;
        lengths.push(totalLen);
    }

    const res = [];
    for (let i = 0; i < totalPoints; i++) {
        const target = (i / totalPoints) * totalLen;
        let idx = lengths.findIndex(l => l >= target);
        if (idx === -1) idx = lengths.length - 1;
        if (idx === 0) { res.push(raw[0]); continue; }
        const r = (target - lengths[idx - 1]) / (lengths[idx] - lengths[idx - 1]);
        res.push({
            x: raw[idx - 1].x + (raw[idx].x - raw[idx - 1].x) * r,
            y: raw[idx - 1].y + (raw[idx].y - raw[idx - 1].y) * r
        });
    }
    return res;
}

export function generatePreset(type) {
    return Array(32).fill(null).map((_, i) => {
        const angle = (i * Math.PI) / 16;
        let x = 0, y = 0;
        if (type === "circle") {
            const r = 0.85; x = r * Math.cos(angle); y = r * Math.sin(angle);
        } else if (type === "heart") {
            const hx = 16 * Math.sin(angle) ** 3;
            const hy = 13 * Math.cos(angle) - 5 * Math.cos(2 * angle) - 2 * Math.cos(3 * angle) - Math.cos(4 * angle);
            const m = 16.5;
            if (i % 2 === 0) { x = hx / m; y = hy / m; }
            else {
                const prev = (i - 1) * Math.PI / 16, next = ((i + 1) % 32) * Math.PI / 16;
                const calc = a => ({ x: 16 * Math.sin(a) ** 3 / m, y: (13 * Math.cos(a) - 5 * Math.cos(2 * a) - 2 * Math.cos(3 * a) - Math.cos(4 * a)) / m });
                const p1 = calc(prev), p2 = calc(next);
                x = (p1.x + p2.x) * 0.5; y = (p1.y + p2.y) * 0.5;
            }
        } else if (type === "star") {
            const outer = Math.floor(i / 2) % 2 === 0;
            const r = i % 2 === 0 ? (outer ? 0.9 : 0.45) : (outer ? 0.75 : 0.55);
            x = r * Math.cos(angle); y = r * Math.sin(angle);
        }
        return { x, y, anchor: i % 2 === 0 };
    });
}
