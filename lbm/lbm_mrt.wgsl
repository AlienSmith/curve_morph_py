// Precomputed LBM constants (uploaded once from CPU)
struct LBMConsts {
  Nx: u32,
  Ny: u32,
  rho0: f32,
  cxs: array<f32, 9>,
  cys: array<f32, 9>,
  weights: array<f32, 9>,
  opp: array<u32, 9>,
  T: array<array<f32, 9>, 9>, // 9x9 MRT transform
  T_inv: array<array<f32, 9>, 9>, // 9x9 inverse MRT transform
  s: array<f32, 9>, // Relaxation parameters
};

// Per-obstacle dynamic data (updated each frame)
struct Obstacle {
  ux: f32,
  uy: f32,
  angular_vel: f32,
  origin: vec2<f32>,
};

@group(0) @binding(0) var<uniform> consts: LBMConsts;
@group(0) @binding(1) var<uniform> obstacles: array<Obstacle, 16>; // Up to 16 dynamic obstacles
@group(0) @binding(2) var F_prev: texture_2d_array<f32>; // 9 layers, read-only previous F
@group(0) @binding(3) var F_curr: texture_storage_2d_array<r32float, write>; // 9 layers, write-only current F
@group(0) @binding(4) var obstacle_mask: texture_2d<u32>; // 0=fluid, ≥1=obstacle ID
@group(0) @binding(5) var velocity_out: texture_storage_2d<rg32float, write>; // Optional: output ux/uy for rendering

@compute @workgroup_size(8, 8)
fn lbm_step(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  if (x >= consts.Nx || y >= consts.Ny) { return; } // Bounds check

  // ---------------------------------------------------------------------------
  // Step 1: Streaming (equivalent to your np.roll calls, but per-cell)
  // ---------------------------------------------------------------------------
  // For each direction i, incoming F comes from (x - cxs[i], y - cys[i]) in the previous step
  var F_streamed: array<f32, 9>;
  @unroll for (var i: u32 = 0u; i < 9u; i++) {
    let sx = (x - u32(consts.cxs[i]) + consts.Nx) % consts.Nx; // Periodic boundaries (adjust for inlet/outlet as needed)
    let sy = (y - u32(consts.cys[i]) + consts.Ny) % consts.Ny;
    F_streamed[i] = textureLoad(F_prev, vec2<i32>(sx, sy), i, 0).x;
  }

  // ---------------------------------------------------------------------------
  // Step 2: Obstacle Boundary Condition
  // ---------------------------------------------------------------------------
  let obstacle_id = textureLoad(obstacle_mask, vec2<i32>(x, y), 0).x;
  if (obstacle_id > 0u) {
    let obs = obstacles[obstacle_id - 1u];
    // Compute local velocity of the obstacle at this cell (for rotating objects)
    let r = vec2<f32>(x, y) - obs.origin;
    let u_solid = vec2<f32>(
      obs.ux - r.y * obs.angular_vel,
      obs.uy + r.x * obs.angular_vel
    );

    // Moving bounceback (Guo-Zheng-Shi boundary condition, standard for dynamic obstacles)
    // Accounts for obstacle velocity to transfer momentum correctly between fluid and solid
    let rho = sum(F_streamed); // Local density for velocity correction
    @unroll for (var i: u32 = 0u; i < 9u; i++) {
      let opp_i = consts.opp[i];
      let cu = consts.cxs[i] * u_solid.x + consts.cys[i] * u_solid.y;
      let F_bounced = F_streamed[opp_i] + 6.0 * consts.weights[i] * rho * cu;
      textureStore(F_curr, vec2<i32>(x, y), i, vec4<f32>(F_bounced));
    }
    velocity_out.store(vec2<i32>(x, y), vec4<f32>(u_solid));
    return;
  }

  // ---------------------------------------------------------------------------
  // Step 3: Compute Macroscopic Variables (rho, ux, uy)
  // ---------------------------------------------------------------------------
  var rho: f32 = 0.0;
  var u: vec2<f32> = vec2<f32>(0.0);
  @unroll for (var i: u32 = 0u; i < 9u; i++) {
    rho += F_streamed[i];
    u.x += F_streamed[i] * consts.cxs[i];
    u.y += F_streamed[i] * consts.cys[i];
  }
  u /= rho;

  // ---------------------------------------------------------------------------
  // Step 4: Compute Equilibrium Distribution (Feq)
  // ---------------------------------------------------------------------------
  var Feq: array<f32, 9>;
  let u_sq = dot(u, u);
  @unroll for (var i: u32 = 0u; i < 9u; i++) {
    let cu = consts.cxs[i] * u.x + consts.cys[i] * u.y;
    Feq[i] = rho * consts.weights[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
  }

  // ---------------------------------------------------------------------------
  // Step 5: MRT Collision (exact match to your Python code)
  // ---------------------------------------------------------------------------
  // Transform to moment space: M = F_streamed @ T.T
  var M: array<f32, 9>;
  @unroll for (var j: u32 = 0u; j < 9u; j++) {
    M[j] = 0.0;
    @unroll for (var i: u32 = 0u; i < 9u; i++) {
      M[j] += F_streamed[i] * consts.T[j][i];
    }
  }

  // Transform Feq to moment space: Meq = Feq @ T.T
  var Meq: array<f32, 9>;
  @unroll for (var j: u32 = 0u; j < 9u; j++) {
    Meq[j] = 0.0;
    @unroll for (var i: u32 = 0u; i < 9u; i++) {
      Meq[j] += Feq[i] * consts.T[j][i];
    }
  }

  // Relax moments
  @unroll for (var j: u32 = 0u; j < 9u; j++) {
    M[j] -= consts.s[j] * (M[j] - Meq[j]);
  }

  // Transform back to distribution space: F_collided = M @ T_inv.T
  var F_collided: array<f32, 9>;
  @unroll for (var i: u32 = 0u; i < 9u; i++) {
    F_collided[i] = 0.0;
    @unroll for (var j: u32 = 0u; j < 9u; j++) {
      F_collided[i] += M[j] * consts.T_inv[i][j];
    }
  }

  // ---------------------------------------------------------------------------
  // Step 6: Write Outputs
  // ---------------------------------------------------------------------------
  @unroll for (var i: u32 = 0u; i < 9u; i++) {
    textureStore(F_curr, vec2<i32>(x, y), i, vec4<f32>(F_collided[i]));
  }
  velocity_out.store(vec2<i32>(x, y), vec4<f32>(u, 0.0, 0.0));
}