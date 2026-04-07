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
    const numSegments = 16;
    const numPoints = 32;

    return Array(numPoints).fill(null).map((_, i) => {
        const angle = (i * Math.PI) / numSegments;
        const isControl = i % 2 !== 0;

        // Scale factor for control points so the quadratic Bezier midpoint
        // exactly touches the anchor radius.
        const angleOffset = Math.PI / numPoints;
        const controlScale = isControl ? (1 / Math.cos(angleOffset)) : 1.0;

        let x = 0, y = 0;

        if (type === "circle") {
            const r = 0.85 * controlScale;
            // Standard Cartesian CCW: Right → Top → Left → Bottom
            x = r * Math.cos(angle);
            y = r * Math.sin(angle);

        } else if (type === "heart") {
            const m = 18; // Divisor to keep it in frame
            // Standard heart traces CW. Negating X mirrors it to CCW.
            const getHeart = (a, s) => {
                const hx = -(16 * Math.sin(a) ** 3);
                const hy = 13 * Math.cos(a) - 5 * Math.cos(2 * a) - 2 * Math.cos(3 * a) - Math.cos(4 * a);
                return { x: (hx / m) * s, y: (hy / m) * s };
            };

            const p = getHeart(angle, controlScale);
            x = p.x;
            y = p.y;

        } else if (type === "star") {
            const isTip = (i % 4 === 0);
            const isPit = ((i - 2) % 4 === 0);

            let r = 0.9;
            if (isPit) r = 0.4;
            if (isControl) r = 0.65; // Mid-way for smoother edges

            // Standard Cartesian CCW: Right → Top → Left → Bottom
            x = r * Math.cos(angle);
            y = r * Math.sin(angle);
        }

        return { x, y, anchor: !isControl };
    });
}
