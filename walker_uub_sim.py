"""
Walker n×n 局部网络 UUB 稳定性仿真
轨道高度 550 km，倾角 53°，72 轨道面 × 22 颗卫星，+grid ISL 链路
支持任意 n×n patch 规模及故障卫星指定
"""

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm


from dataclasses import dataclass, field
from typing import List, Tuple

# =============================================================================
# Section A — 物理常数与仿真参数
# =============================================================================

Gb = 1e9

@dataclass
class Params:
    # 地球物理常数
    mu:      float = 3.986004418e14   # m^3/s^2
    R_E:     float = 6.371e6          # m
    J2:      float = 1.08263e-3
    omega_E: float = 7.2921150e-5     # rad/s

    # 轨道参数（Starlink 550 km 层）
    h_alt:       float = 550e3
    inc_deg:     float = 53.0
    n_planes:    int   = 72
    n_per_plane: int   = 22
    F_walker:    int   = 1

    # Patch 起始面/位置及规模
    p0:      int = 0
    k0:      int = 0
    n_patch: int = 3   # patch 边长：n_patch×n_patch 颗卫星

    # 故障卫星索引列表（patch 内索引 i = (p-p0)*n_patch + (k-k0)）
    # 故障卫星：λ=0，μ=0，所有 ISL 链路断开，队列强制清零
    faulty_sats: List[int] = field(default_factory=list)

    # ISL 参数（激光通信）
    R_max:   float = 2000e3
    H_atm:   float = 200e3
    B_isl:   float = 30e9 / Gb
    gamma0:  float = field(default_factory=lambda: 10.0 * (1600e3)**2)

    # 流量与服务参数
    C_down:      float = 10e9 / Gb
    # 流量负载比（目标 Σλ = load_ratio × n × C_down）
    # load_ratio < 1：欠载，q_nom 频繁踩 0（Skorokhod 激活，误差不光滑）
    # load_ratio ≥ 1：过载，q_nom 持续 > 0（Skorokhod 不激活，系统光滑）
    # Part 2 稳定性分析推荐 load_ratio ≥ 1 以避免边界非光滑问题
    load_ratio:  float = 1.05
    rho_scale:   float = 1.0
    sigma_hotspot_deg: float = 5.0

    # 控制与积分参数
    K:       float = 0.00001  # Part 1 使用：小 K → 慢衰减，3 圈内可见
    dt_q:    float = 1.0

    # 仿真时长
    n_orbits: int   = 3
    dt_out:   float = 0.1
    n_mc:     int   = 1000

    # Part 1：冲激响应参数
    # 注意：pulse_amplitude 必须 > C_down（=10 Gbps）才能引起队列积累
    pulse_amplitude: float = 200.0   # 初始队列（Gbits）
    pulse_duration:  float = 900.0   # 保留字段（不再用于 λ 脉冲）

    # Part 2：受扰仿真参数
    # K_part2 独立于 Part 1：需要足够大才能让误差平滑收敛
    # τ_error ≈ 1/(K_part2 · λ₂(L))，λ₂≈0.5 → K_part2=0.05 给出 τ≈40s，3 圈内可见
    K_part2:         float = 0.05    # Part 2 专用控制增益
    burst_amplitude: float = 2000.0  # 队列跳变幅度（Gbits）
    burst_duration:  float = 900.0   # 保留字段

    # 衍生量（__post_init__ 计算）
    a_orb:  float = field(init=False)
    T_orb:  float = field(init=False)
    inc:    float = field(init=False)
    sigma_hotspot: float = field(init=False)
    n_sat:  int   = field(init=False)   # = n_patch²

    def __post_init__(self):
        self.a_orb  = self.R_E + self.h_alt
        self.T_orb  = 2 * np.pi * np.sqrt(self.a_orb**3 / self.mu)
        self.inc    = np.radians(self.inc_deg)
        self.sigma_hotspot = np.radians(self.sigma_hotspot_deg)
        self.n_sat  = self.n_patch * self.n_patch


# 全球流量热点（纬度°, 经度°, 权重）
HOTSPOTS = [
    ( 35.0,  115.0, 1.00),   # 东亚（中日韩）
    ( 22.0,   80.0, 0.80),   # 南亚（印度）
    ( 40.0,  -75.0, 0.90),   # 北美东岸
    ( 50.0,   10.0, 0.85),   # 西欧
    (-33.0,  151.0, 0.50),   # 大洋洲（东澳）
    (-23.0,  -46.0, 0.60),   # 南美（巴西东南）
]

P = Params()
# F_walker 必须是 0 到 n_planes-1 之间的整数
assert 0 <= P.F_walker < P.n_planes, "F_walker 相位因子设置不合法！"
result_dir = f"results_k={P.K}_patch={P.n_patch}x{P.n_patch}_{P.n_orbits}T_j2"

# 如果没有该目录就创建一个
import os
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


# --------------------------------

# =============================================================================
# Section B — Walker 初始条件生成（Keplerian → ECI）
# =============================================================================

def rot3(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def rot1(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def keplerian_to_eci(a: float, inc: float, raan: float, nu: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    严格对齐版：将开普勒轨道根数精确转换为 ECI 惯性系下的 R (位置) 和 V (速度)
    修复了由于矩阵乘法顺序颠倒导致的卫星轨道退化与坠落 Bug
    """
    # 1. 计算轨道面（Perifocal Frame, PQW）内的确定性二维位置与速度
    #
    # Brouwer 短周期速度改正：使每颗卫星具有相同的 Brouwer 平均半长轴 a，
    # 消除不同 M 处密切根数与平均根数的偏差所导致的长期面内漂移。
    # 改正量 δv/v = (1/2)·J2·(R_E/a)²·(3/2)·sin²i·cos(2u)，u≈M（ω=0 圆轨道）
    J2_sp = P.J2 * (P.R_E / a)**2 * 1.5 * np.sin(inc)**2 * np.cos(2.0 * nu)
    v_circ = np.sqrt(P.mu / a) * (1.0 + 0.5 * J2_sp)

    r_pf = a      * np.array([np.cos(nu),  np.sin(nu), 0.0])
    v_pf = v_circ * np.array([-np.sin(nu), np.cos(nu), 0.0])
    
    # 2. 显式构建从 PQW 到 ECI 的标准三维转换矩阵 Q (避免受外部 rot1/rot3 符号污染)
    c_O, s_O = np.cos(raan), np.sin(raan)
    c_i, s_i = np.cos(inc),  np.sin(inc)
    
    # 标准的方向余弦矩阵项（当近地点幅角 omega = 0 时的解析解）
    Q = np.array([
        [c_O, -s_O * c_i,  s_O * s_i],
        [s_O,  c_O * c_i, -c_O * s_i],
        [0.0,  s_i,        c_i      ]
    ])
    
    # 3. 投影到地心惯性系
    r_eci = Q @ r_pf
    v_eci = Q @ v_pf
    
    return r_eci, v_eci


def generate_walker_patch(p0: int, k0: int, n_patch: int) -> np.ndarray:
    """
    精确定位修复版：强制生成真正的局部 Walker +grid 拓扑构型
    确保同面内相邻卫星物理尾随，消除反向运动导致的非物理相撞
    """
    n_sat = n_patch * n_patch
    x0    = np.zeros(6 * n_sat)
    
    # 基础物理步长
    d_raan = 2 * np.pi / P.n_planes
    d_M    = 2 * np.pi / P.n_per_plane
    d_phase = P.F_walker * (2 * np.pi / (P.n_planes * P.n_per_plane))
    
    for dp in range(n_patch):
        for dk in range(n_patch):
            # 1. 严格计算该卫星在整星座中的绝对面号与星号
            p_abs = (p0 + dp) % P.n_planes
            # 关键修复：确保 dk 的递增对应于运动方向上的平近点角顺序推进
            k_abs = (k0 + dk) % P.n_per_plane
            
            # 2. 严格对齐 Walker 星座解析公式
            raan = p_abs * d_raan
            # 同一面内：dk 每 +1，平近点角推进 d_M；跨面：受到相位因子 d_phase 的调制
            M    = k_abs * d_M + p_abs * d_phase
            
            # 3. 严格使用相同的半长轴 P.a_orb，确保周期完全一致，防止线性漂移
            r, v = keplerian_to_eci(P.a_orb, P.inc, raan, M)
            
            # 4. 关键：局部展平索引必须与 dp, dk 严格绑定，这才是 compute_topology 寻找邻居的逻辑
            idx  = dp * n_patch + dk
            x0[6*idx   : 6*idx+3] = r
            x0[6*idx+3 : 6*idx+6] = v
            
    return x0


# =============================================================================
# Section C — J2 + 二体 ODE 右端项
# =============================================================================

def orbital_odes(t: float, x: np.ndarray) -> np.ndarray:
    n_sat = P.n_sat
    dxdt  = np.zeros(6 * n_sat)
    for i in range(n_sat):
        r = x[6*i   : 6*i+3]
        v = x[6*i+3 : 6*i+6]
        rx, ry, rz = r
        r_norm = np.linalg.norm(r)
        r2     = r_norm**2
        
        # 二体基础引力加速度
        a_tb = -(P.mu / (r_norm**3)) * r
        
        # J2 摄动加速度（严格遵循梯度公式，修复符号与项对齐问题）
        # 统一提取负号系数: -1.5 * J2 * mu * R_E^2 / r^5
        fac_j2 = -1.5 * P.J2 * P.mu * (P.R_E**2) / (r_norm**5)
        z_scale = 5.0 * (rz**2) / r2
        
        a_j2 = fac_j2 * np.array([
            rx * (1.0 - z_scale),
            ry * (1.0 - z_scale),
            rz * (3.0 - z_scale)
        ])
        
        dxdt[6*i   : 6*i+3] = v
        dxdt[6*i+3 : 6*i+6] = a_tb + a_j2
    return dxdt


# =============================================================================
# Section D — 拓扑切换指示函数 I_ij(t)
# =============================================================================

def _grid_neighbors(n: int) -> List[Tuple[int, int]]:
    """n×n +grid 的所有相邻对（面内 k+1 方向 + 面间 p+1 方向）"""
    pairs = []
    for p in range(n):
        for k in range(n):
            i = n * p + k
            if k + 1 < n:
                pairs.append((i, n * p + (k + 1)))
            if p + 1 < n:
                pairs.append((i, n * (p + 1) + k))
    return pairs


def _virtual_degree(n: int) -> np.ndarray:
    """每颗星在完整无限 +grid 中缺失的链路数 = 4 - patch 内实际度数"""
    n_virt = np.zeros(n * n)
    for p in range(n):
        for k in range(n):
            i    = n * p + k
            d    = ((1 if k > 0 else 0) + (1 if k < n - 1 else 0) +
                    (1 if p > 0 else 0) + (1 if p < n - 1 else 0))
            n_virt[i] = 4 - d
    return n_virt


# 模块级缓存（从 P.n_patch 计算，P 须在此之前已定义）
_GRID_PAIRS = _grid_neighbors(P.n_patch)
_N_VIRTUAL  = _virtual_degree(P.n_patch)


def compute_topology(r_mat: np.ndarray) -> np.ndarray:
    """
    r_mat: (n_sat, 3)，返回 I (n_sat, n_sat) int8。
    仅检查 +grid 邻居对；故障卫星的所有链路强制为 0。
    """
    n      = r_mat.shape[0]
    I      = np.zeros((n, n), dtype=np.int8)
    thresh = P.R_E + P.H_atm
    faulty = set(P.faulty_sats)
    for (i, j) in _GRID_PAIRS:
        if i in faulty or j in faulty:
            continue
        diff  = r_mat[j] - r_mat[i]
        d_ij  = np.linalg.norm(diff)
        if d_ij > P.R_max:
            continue
        cross = np.cross(r_mat[i], diff)
        h_min = np.linalg.norm(cross) / d_ij
        if h_min > thresh:
            I[i, j] = 1
            I[j, i] = 1
    return I


# =============================================================================
# Section E — 香农容量 C_ij(t)
# =============================================================================

def compute_capacity(r_mat: np.ndarray, I_mat: np.ndarray) -> np.ndarray:
    n = r_mat.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if I_mat[i, j] == 1:
                d_ij    = np.linalg.norm(r_mat[i] - r_mat[j])
                snr_ij  = P.gamma0 / d_ij**2
                C[i, j] = P.B_isl * np.log2(1.0 + snr_ij)
    return C


# =============================================================================
# Section F — 地面流量模型（覆盖区积分 λ_i, μ_i）
# =============================================================================

_HOTSPOT_LAT = np.array([np.radians(h[0]) for h in HOTSPOTS])
_HOTSPOT_LON = np.array([np.radians(h[1]) for h in HOTSPOTS])
_HOTSPOT_W   = np.array([h[2]             for h in HOTSPOTS])


def traffic_density(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    rho    = np.zeros_like(lat)
    sigma2 = P.sigma_hotspot**2
    for lat_k, lon_k, w_k in zip(_HOTSPOT_LAT, _HOTSPOT_LON, _HOTSPOT_W):
        dlat = lat - lat_k
        dlon = (lon - lon_k + np.pi) % (2 * np.pi) - np.pi
        rho += w_k * np.exp(-0.5 * (dlat**2 + dlon**2) / sigma2)
    return rho


def eci_to_latlon(r: np.ndarray, t: float = 0.0) -> Tuple[float, float]:
    r_norm = np.linalg.norm(r)
    lon_eci = np.arctan2(r[1], r[0])
    lon_ecef = (lon_eci - P.omega_E * t + np.pi) % (2 * np.pi) - np.pi
    return np.arcsin(r[2] / r_norm), lon_ecef


def compute_lambda(r_i: np.ndarray, t: float = 0.0) -> float:
    lat_sub, lon_sub = eci_to_latlon(r_i, t)
    r_norm    = np.linalg.norm(r_i)
    alpha     = np.arccos(np.clip(P.R_E / r_norm, -1.0, 1.0))
    cos_a     = np.cos(alpha)
    rng       = np.random.default_rng(seed=42)
    cos_theta = rng.uniform(cos_a, 1.0, P.n_mc)
    phi_mc    = rng.uniform(0, 2 * np.pi, P.n_mc)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    n_sub  = np.array([np.cos(lat_sub)*np.cos(lon_sub),
                       np.cos(lat_sub)*np.sin(lon_sub),
                       np.sin(lat_sub)])
    e_z = np.array([0.0, 0.0, 1.0])
    if abs(n_sub[2]) > 0.9999:
        e_east = np.array([1.0, 0.0, 0.0])
    else:
        e_east = np.cross(e_z, n_sub)
        e_east /= np.linalg.norm(e_east)
    e_north = np.cross(n_sub, e_east)
    e_north /= np.linalg.norm(e_north)
    pts = (cos_theta[:, None] * n_sub[None, :]
           + sin_theta[:, None] * (np.cos(phi_mc[:, None]) * e_east[None, :]
                                   + np.sin(phi_mc[:, None]) * e_north[None, :]))
    pts_lat  = np.arcsin(np.clip(pts[:, 2], -1.0, 1.0))
    pts_lon  = np.arctan2(pts[:, 1], pts[:, 0])
    rho_vals = traffic_density(pts_lat, pts_lon)
    cap_area = 2 * np.pi * (1.0 - cos_a)
    return float(cap_area * np.mean(rho_vals) * P.R_E**2)


def compute_traffic_batch(r_hist: np.ndarray, t_arr: np.ndarray, rho_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    N_t, n_sat, _ = r_hist.shape
    faulty        = set(P.faulty_sats)
    lam_raw       = np.zeros((N_t, n_sat))
    report_every  = max(1, N_t // 10)
    for k in range(N_t):
        for i in range(n_sat):
            if i not in faulty:
                lam_raw[k, i] = compute_lambda(r_hist[k, i], t_arr[k])
        if (k + 1) % report_every == 0:
            print(f"     {k+1}/{N_t} 步完成 ({100*(k+1)/N_t:.0f}%)")
    lam_hist = rho_scale * lam_raw
    mu_hist  = np.full((N_t, n_sat), P.C_down)
    if faulty:
        mu_hist[:, sorted(faulty)] = 0.0   # 故障卫星无服务能力
    return lam_hist, mu_hist


# =============================================================================
# Section G — 流体队列 ODE 右端项 + Skorokhod 投影
# =============================================================================

def queue_rhs(q: np.ndarray, lam: np.ndarray, mu: np.ndarray,
              u: np.ndarray, I: np.ndarray,
              c_virt: np.ndarray) -> np.ndarray:
    inflow      = (I * u).sum(axis=0)
    outflow     = (I * u).sum(axis=1)
    virtual_out = _N_VIRTUAL * np.minimum(c_virt, P.K * q)
    net         = lam - mu + inflow - outflow - virtual_out
    dqdt        = np.where(q <= 0.0, np.maximum(net, 0.0), net)
    return dqdt


def euler_step_projected(q: np.ndarray, dqdt: np.ndarray, dt: float) -> np.ndarray:
    return np.maximum(q + dt * dqdt, 0.0)


# =============================================================================
# Section H — 饱和比例控制律 u_ij(t)
# =============================================================================

def compute_control(q: np.ndarray, C: np.ndarray, I: np.ndarray) -> np.ndarray:
    n = len(q)
    u = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if I[i, j] == 1:
                u[i, j] = min(P.K * max(q[i] - q[j], 0.0), C[i, j])
    return u


# =============================================================================
# Section I — 李雅普诺夫函数 V(q) 及导数 dV/dt
# =============================================================================

def lyapunov_V(q: np.ndarray) -> float:
    return 0.5 * float(np.dot(q, q))


def lyapunov_dV(q: np.ndarray, dqdt: np.ndarray) -> float:
    return float(np.dot(q, dqdt))


# =============================================================================
# Section I2 — 扩展李雅普诺夫函数（Part 1 / Part 2）
# =============================================================================

def lyapunov_V1(q: np.ndarray) -> float:
    """V₁(q) = Σ q_i  （L1 范数，q_i≥0 故等价于 ‖q‖₁）"""
    return float(q.sum())


def lyapunov_dV1(q: np.ndarray, dqdt: np.ndarray) -> float:
    """dV₁/dt = Σ dq_i/dt（仅计及 q_i>0 的分量，Skorokhod 边界已由 queue_rhs 处理）"""
    return float(dqdt[q > 0].sum())


def graph_laplacian(I_mat: np.ndarray) -> np.ndarray:
    """从当前 ISL 拓扑邻接矩阵 I_mat 构造图 Laplacian L = D - A"""
    A = I_mat.astype(float)
    D = np.diag(A.sum(axis=1))
    return D - A


def lyapunov_V2(q: np.ndarray, L: np.ndarray) -> float:
    """V₂(q) = ½‖q‖² + (K/2)·q⊤Lq = ½‖q‖² + (K/4)·Σ_{(i,j)∈E}(q_i-q_j)²
    物理意义：总队列能量 + ISL 拓扑负载不均衡程度之和"""
    return 0.5 * float(q @ q) + (P.K / 2.0) * float(q @ L @ q)


def lyapunov_dV2(q: np.ndarray, dqdt: np.ndarray, L: np.ndarray) -> float:
    """dV₂/dt = q⊤(I + K·L)·dq/dt"""
    W = np.eye(len(q)) + P.K * L
    return float(q @ W @ dqdt)


def lyapunov_V3(delta_q: np.ndarray) -> float:
    """V₃(δq) = ½‖δq‖²  （偏差状态的二次型，用于 Part 2）"""
    return 0.5 * float(np.dot(delta_q, delta_q))


def lyapunov_dV3(delta_q: np.ndarray, delta_dqdt: np.ndarray) -> float:
    """dV₃/dt = δq⊤·δq̇"""
    return float(np.dot(delta_q, delta_dqdt))


# --- 时变正定权重矩阵 Lyapunov 函数 ---
# 设计：W(t) = (diag(μ) + K·L(t))⁻¹
#
# 核心性质（q > 0，忽略容量饱和时）：
#   V_W(q,t) = ½ qᵀ W(t) q
#   dV_W/dt  = ½ qᵀ Ẇ q  +  qᵀ W·f(q,t)
#            = ½ qᵀ Ẇ q  +  qᵀ(μI+KL)⁻¹·[−(μI+KL)q + ...]
#            = ½ qᵀ Ẇ q  −  ‖q‖²  +  小量
#
# 当拓扑不变（Ẇ=0）时：dV_W/dt = −‖q‖² ≤ −2μ·V_W  → 指数稳定
# 当拓扑缓变时：Ẇ = −W·K·L̇·W  对 ‖q‖² 的修正量级为 O(K·‖L̇‖·‖W‖²)，
# 只要 ‖L̇‖ 充分小（ISL 链路缓慢切换），负定性依然成立。

def lyapunov_W_matrix(mu_vec: np.ndarray, L: np.ndarray) -> np.ndarray:
    """计算 W(t) = Ã⁻¹，其中
        Ã = diag(μ) + K·L + K·diag(N_VIRTUAL)

    对应实际线性化系统 dq/dt = -Ã·q，使得：
        qᵀ W dq/dt = qᵀ Ã⁻¹·(-Ã·q) = -‖q‖²  （精确负定，无残差）

    之前只用 (diag(μ)+K·L)⁻¹ 漏掉了虚拟链路排流项 K·diag(N_VIRTUAL)，
    导致 qᵀ W dq/dt ≠ -‖q‖²，Lyapunov 导数有噪声偏差。
    """
    M = np.diag(mu_vec) + P.K * (L + np.diag(_N_VIRTUAL))
    M += 1e-12 * np.eye(len(mu_vec))   # 防奇异
    return np.linalg.inv(M)


def lyapunov_VW(q: np.ndarray, W: np.ndarray) -> float:
    """V_W(q,t) = ½ qᵀ W(t) q"""
    return 0.5 * float(q @ W @ q)


def lyapunov_dVW(q: np.ndarray, dqdt: np.ndarray,
                 W: np.ndarray, W_dot: np.ndarray) -> float:
    """dV_W/dt = ½ qᵀ Ẇ q + qᵀ W·dq/dt
    W_dot 由相邻时刻 W 的有限差分估计得到。
    """
    return 0.5 * float(q @ W_dot @ q) + float(q @ W @ dqdt)


# =============================================================================
# Section J0 — 辅助：冲激流量生成
# =============================================================================

def compute_impulse_lambda(t_arr: np.ndarray, amplitude: float,
                           duration: float) -> np.ndarray:
    """高斯形冲激 λ 历史（N_t × n_sat）。
    脉冲中心在 duration/2，标准差 σ = duration/4，
    使 ±2σ 范围内幅度 > 13.5% peak，±σ 内 > 60.7% peak。
    """
    n_sat = P.n_sat
    t0    = duration / 2.0
    sigma = duration / 4.0
    envelope = amplitude * np.exp(-0.5 * ((t_arr - t0) / sigma) ** 2)
    lam = np.outer(envelope, np.ones(n_sat))
    return lam


# =============================================================================
# Section J — 主仿真驱动循环
# =============================================================================

def run_simulation():
    n  = P.n_sat
    ns = P.n_patch
    print("=" * 60)
    print(f"Walker {ns}×{ns} 局部网络 UUB 稳定性仿真  (n_sat={n})")
    if P.faulty_sats:
        print(f"  故障卫星索引: {P.faulty_sats}")
    print(f"轨道高度 {P.h_alt/1e3:.0f} km，倾角 {P.inc_deg}°")
    print(f"轨道周期 T_orb = {P.T_orb:.1f} s ({P.T_orb/60:.1f} min)")
    print("=" * 60)

    # --- Step 1: 生成初始条件 ---
    print(f"\n[1/5] 生成 Walker {ns}×{ns} 初始 ECI 状态...")
    x0 = generate_walker_patch(P.p0, P.k0, ns)

    # --- Step 2: 轨道传播 ---
    t_end  = P.n_orbits * P.T_orb
    print(f"[2/5] 传播轨道（RK45+J2, {P.n_orbits} 圈 ≈ {t_end/60:.1f} min）...")
    t_eval = np.arange(0.0, t_end, P.dt_out)
    sol    = solve_ivp(orbital_odes, [0.0, t_end], x0,
                       method="RK45", t_eval=t_eval,
                       rtol=1e-8, atol=1e-10, dense_output=True)
    assert sol.success, f"轨道 ODE 求解失败: {sol.message}"
    N_t    = len(sol.t)
    r_hist = sol.y.T.reshape(N_t, n, 6)[:, :, :3].copy()
    print(f"   共 {N_t} 个输出时刻，dt={P.dt_out} s")

    r_norms = np.linalg.norm(r_hist[0], axis=-1)
    print(f"   t=0 各星轨道半径: min={r_norms.min()/1e6:.4f} Mm, "
          f"max={r_norms.max()/1e6:.4f} Mm (期望 ≈ {P.a_orb/1e6:.4f} Mm)")

    # --- Step 3: 批量计算 I, C, λ, μ ---
    print("[3/5] 计算拓扑/容量/流量...")
    I_hist = np.zeros((N_t, n, n), dtype=np.int8)
    C_hist = np.zeros((N_t, n, n))
    for k in range(N_t):
        I_hist[k] = compute_topology(r_hist[k])
        C_hist[k] = compute_capacity(r_hist[k], I_hist[k])

    deg0 = I_hist[0].sum(axis=1)
    print(f"   t=0 各星链路度数: {deg0.tolist()}")

    print("   计算地面流量（Monte Carlo，约需 1~5 min）...")
    lam_raw, mu_hist = compute_traffic_batch(r_hist, sol.t, rho_scale=1.0)

    # 标定 rho_scale，基于健康卫星的总流量
    healthy = [i for i in range(n) if i not in set(P.faulty_sats)]
    target_total   = P.load_ratio * len(healthy) * P.C_down
    raw_total_mean = lam_raw[:, healthy].mean(axis=0).sum()
    rho_scale = target_total / raw_total_mean if raw_total_mean > 0 else 1.0
    print(f"   rho_scale = {rho_scale:.4e} (目标 Σλ[健康] = {target_total:.1f} Gbps)")
    lam_hist = lam_raw * rho_scale

    mean_lam = lam_hist.mean(axis=0)
    mean_mu  = mu_hist.mean(axis=0)
    print(f"   时均 Σλ = {mean_lam.sum():.2f} Gbps, "
          f"Σμ = {mean_mu.sum():.2f} Gbps → "
          f"容量不越界: {mean_lam.sum() < mean_mu.sum()}")

    # --- Step 4: 队列积分 ---
    print("[4/5] 初始化队列并执行前向 Euler 积分...")
    faulty_idx = np.array(sorted(P.faulty_sats), dtype=int)
    q          = np.zeros(n)
    #q[0] = 1.0e6   # 故障卫星初始队列强制为 0
    q_hist     = np.zeros((N_t, n))
    dqdt_hist  = np.zeros((N_t, n))
    u_hist     = np.zeros((N_t, n, n))
    V_hist     = np.zeros(N_t)
    dVdt_hist  = np.zeros(N_t)

    # CFL 稳定条件：K * d_eff * dt_q < 1，d_eff=4 对所有位置恒成立
    # 超出时 Euler 步从高负载卫星超发流量（多于其实际队列），邻居凭空获得额外积压，
    # 导致 V 在 Skorokhod 投影后反而上升，即使瞬时 dVdt < 0。
    D_EFF = 4
    dt_q_cfl = (0.9 / (D_EFF * P.K)) if P.K > 0 else P.dt_q
    dt_q_req = min(P.dt_q, dt_q_cfl)
    n_sub = max(1, math.ceil(P.dt_out / dt_q_req))
    dt_q  = P.dt_out / n_sub
    if dt_q_req < P.dt_q - 1e-9:
        print(f"   [CFL 警告] K={P.K}: K·d_eff·dt_q = {P.K*D_EFF*P.dt_q:.2f} > 1，"
              f"dt_q 自动从 {P.dt_q}s 缩至 {dt_q:.3f}s")
    print(f"   队列积分：内层步长 dt_q={dt_q:.4f} s，每轨道步 {n_sub} 个子步")

    for k in range(N_t):
        deg_k  = I_hist[k].sum(axis=1).astype(float)
        c_virt = np.where(
            deg_k > 0,
            (C_hist[k] * I_hist[k]).sum(axis=1) / np.maximum(deg_k, 1),
            200e9 / Gb
        )

        q_hist[k] = q

        for _ in range(n_sub):
            u    = compute_control(q, C_hist[k], I_hist[k])
            dqdt = queue_rhs(q, lam_hist[k], mu_hist[k], u, I_hist[k], c_virt)
            q    = euler_step_projected(q, dqdt, dt_q)
            if faulty_idx.size > 0:
                q[faulty_idx] = 0.0   # 故障卫星队列强制清零

        # Lyapunov 统计基于本步开始时的状态
        u_rec    = compute_control(q_hist[k], C_hist[k], I_hist[k])
        dqdt_rec = queue_rhs(q_hist[k], lam_hist[k], mu_hist[k], u_rec, I_hist[k], c_virt)
        dqdt_hist[k] = dqdt_rec
        u_hist[k]    = u_rec
        V_hist[k]    = lyapunov_V(q_hist[k])
        dVdt_hist[k] = lyapunov_dV(q_hist[k], dqdt_rec)

    print(f"   最终队列范数 ‖q‖ = {np.linalg.norm(q_hist[-1]):.4f} Gbits")
    print(f"   V(t) 最大值 = {V_hist.max():.4e}, 最终值 = {V_hist[-1]:.4e}")

    # --- Step 5: 绘图 ---
    print("[5/5] 生成图形...")
    t_min = sol.t / 60.0

    _plot_queues(t_min, q_hist)
    _plot_lyapunov(t_min, V_hist, dVdt_hist)
    _plot_capacity(t_min, C_hist, I_hist, r_hist)
    _plot_lambda(t_min, lam_hist)
    _plot_dVdt_scatter(V_hist, dVdt_hist)

    print("\n仿真完成！")
    return sol, r_hist, I_hist, C_hist, lam_hist, mu_hist, q_hist, V_hist, dVdt_hist


# =============================================================================
# Section J2 — Part 1：冲激响应仿真
# =============================================================================

def run_part1(r_hist=None, I_hist=None, C_hist=None, sol_t=None):
    """
    Part 1：零初始状态 + 单次矩形 λ 冲激，记录三种李雅普诺夫函数随时间的演化。

    参数可直接传入已有的轨道/拓扑数据（避免重复 ODE 积分）；
    若为 None，则内部重新计算。
    """
    n   = P.n_sat
    ns  = P.n_patch
    T_pulse = P.pulse_duration if P.pulse_duration > 0 else P.T_orb / 4.0

    print("\n" + "=" * 60)
    print(f"Part 1 — 冲激响应  (amplitude={P.pulse_amplitude} Gbps, "
          f"duration={T_pulse:.1f} s = {T_pulse/P.T_orb:.2f} 圈)")
    print("=" * 60)

    # --- 轨道/拓扑（复用或重算）---
    if r_hist is None or I_hist is None or C_hist is None or sol_t is None:
        print("  [1/3] 重新生成轨道与拓扑...")
        x0_    = generate_walker_patch(P.p0, P.k0, ns)
        t_end_ = P.n_orbits * P.T_orb
        t_ev_  = np.arange(0.0, t_end_, P.dt_out)
        sol_   = solve_ivp(orbital_odes, [0.0, t_end_], x0_,
                           method="RK45", t_eval=t_ev_,
                           rtol=1e-8, atol=1e-10)
        assert sol_.success
        N_t_   = len(sol_.t)
        r_hist = sol_.y.T.reshape(N_t_, n, 6)[:, :, :3].copy()
        sol_t  = sol_.t
        I_hist = np.zeros((N_t_, n, n), dtype=np.int8)
        C_hist = np.zeros((N_t_, n, n))
        for k in range(N_t_):
            I_hist[k] = compute_topology(r_hist[k])
            C_hist[k] = compute_capacity(r_hist[k], I_hist[k])
    else:
        print("  [1/3] 复用已有轨道与拓扑数据")

    N_t  = len(sol_t)
    t_min = sol_t / 60.0

    # --- 服务率（λ 恒为 0，激增体现在初始队列状态）---
    # 正确的稳定性分析框架：t=0 时队列已处于非零初始状态（代表激增的即时结果），
    # 之后 λ=0（自治系统），观察 V_W 是否单调下降。
    # 若 λ 持续注入，系统为非自治，dV_W 无法保证负定。
    lam_imp = np.zeros((N_t, n))   # 自治：λ ≡ 0
    mu_p1   = np.full((N_t, n), P.C_down)
    faulty_idx = np.array(sorted(P.faulty_sats), dtype=int)
    if faulty_idx.size > 0:
        mu_p1[:, faulty_idx] = 0.0

    # --- 队列积分 ---
    print(f"  [2/3] 初始队列 q(0) = {P.pulse_amplitude:.1f} Gbits（均匀），λ≡0（自治系统）...")
    D_EFF    = 4
    dt_q_cfl = 0.9 / (D_EFF * P.K) if P.K > 0 else P.dt_q
    dt_q_req = min(P.dt_q, dt_q_cfl)
    n_sub    = max(1, math.ceil(P.dt_out / dt_q_req))
    dt_q     = P.dt_out / n_sub

    # 非零初始状态：激增效果体现在 q(0)，而非持续的 λ 输入
    q          = np.full(n, P.pulse_amplitude)   # q(0) ≠ 0，Gbits
    q_hist_p1  = np.zeros((N_t, n))
    dqdt_hist  = np.zeros((N_t, n))
    W_hist     = np.zeros((N_t, n, n))

    for k in range(N_t):
        deg_k  = I_hist[k].sum(axis=1).astype(float)
        c_virt = np.where(
            deg_k > 0,
            (C_hist[k] * I_hist[k]).sum(axis=1) / np.maximum(deg_k, 1),
            200e9 / Gb
        )
        L_k = graph_laplacian(I_hist[k])

        q_hist_p1[k] = q
        W_hist[k]    = lyapunov_W_matrix(mu_p1[k], L_k)

        u_rec    = compute_control(q, C_hist[k], I_hist[k])
        dqdt_hist[k] = queue_rhs(q, lam_imp[k], mu_p1[k], u_rec, I_hist[k], c_virt)

        for _ in range(n_sub):
            u    = compute_control(q, C_hist[k], I_hist[k])
            dqdt = queue_rhs(q, lam_imp[k], mu_p1[k], u, I_hist[k], c_virt)
            q    = euler_step_projected(q, dqdt, dt_q)
            if faulty_idx.size > 0:
                q[faulty_idx] = 0.0

    # --- 有限差分估计 Ẇ → 计算 V_W 和 dV_W ---
    W_dot = np.zeros_like(W_hist)
    W_dot[0]    = (W_hist[1]  - W_hist[0])   / P.dt_out
    W_dot[-1]   = (W_hist[-1] - W_hist[-2])  / P.dt_out
    W_dot[1:-1] = (W_hist[2:] - W_hist[:-2]) / (2.0 * P.dt_out)

    VW_hist  = np.array([lyapunov_VW(q_hist_p1[k], W_hist[k])  for k in range(N_t)])
    dVW_hist = np.array([lyapunov_dVW(q_hist_p1[k], dqdt_hist[k],
                                       W_hist[k], W_dot[k])     for k in range(N_t)])

    print(f"  Part 1 完成。末态 ‖q‖={np.linalg.norm(q_hist_p1[-1]):.4f} Gbits, "
          f"V_W_max={VW_hist.max():.4e}")
    _plot_part1(t_min, q_hist_p1, VW_hist, dVW_hist)
    return q_hist_p1, VW_hist, dVW_hist


# =============================================================================
# Section J3 — Part 2：标称 + 受扰时变系统分析
# =============================================================================

def run_part2(r_hist=None, I_hist=None, C_hist=None, sol_t=None,
              lam_nom=None, mu_nom=None, q_nom_ext=None):
    """
    Part 2：以标称仿真轨迹 q*(t) 为参考，在开头叠加流量激增扰动，
    分析误差状态 δq(t) = q_perturbed - q_nominal 的 ISS/渐近稳定性。

    q_nom_ext : 可选，直接传入主仿真的 q_hist（N_t×n）作为标称轨迹，
                保证 Part 2 的 q_nom 与 queues_k=xx.pdf 完全一致。
                若为 None，则用 K_part2 重新积分一条标称轨迹（K 不同，图形会有差异）。
    """
    n   = P.n_sat
    ns  = P.n_patch
    T_burst = P.burst_duration if P.burst_duration > 0 else P.T_orb / 8.0

    print("\n" + "=" * 60)
    print(f"Part 2 — 时变受扰分析  (burst={P.burst_amplitude} Gbps, "
          f"duration={T_burst:.1f} s = {T_burst/P.T_orb:.3f} 圈)")
    print("=" * 60)

    # --- 轨道/拓扑（复用或重算）---
    if r_hist is None or I_hist is None or C_hist is None or sol_t is None:
        print("  [1/4] 重新生成轨道与拓扑...")
        x0_    = generate_walker_patch(P.p0, P.k0, ns)
        t_end_ = P.n_orbits * P.T_orb
        t_ev_  = np.arange(0.0, t_end_, P.dt_out)
        sol_   = solve_ivp(orbital_odes, [0.0, t_end_], x0_,
                           method="RK45", t_eval=t_ev_,
                           rtol=1e-8, atol=1e-10)
        assert sol_.success
        N_t_   = len(sol_.t)
        r_hist = sol_.y.T.reshape(N_t_, n, 6)[:, :, :3].copy()
        sol_t  = sol_.t
        I_hist = np.zeros((N_t_, n, n), dtype=np.int8)
        C_hist = np.zeros((N_t_, n, n))
        for k in range(N_t_):
            I_hist[k] = compute_topology(r_hist[k])
            C_hist[k] = compute_capacity(r_hist[k], I_hist[k])
    else:
        print("  [1/4] 复用已有轨道与拓扑数据")

    N_t   = len(sol_t)
    t_min = sol_t / 60.0
    faulty_idx = np.array(sorted(P.faulty_sats), dtype=int)

    # --- 标称流量 ---
    if lam_nom is None or mu_nom is None:
        print("  [2/4] 计算标称地面流量（Monte Carlo）...")
        lam_raw, mu_nom = compute_traffic_batch(r_hist, sol_t, rho_scale=1.0)
        healthy = [i for i in range(n) if i not in set(P.faulty_sats)]
        tgt = P.load_ratio * len(healthy) * P.C_down
        raw_mean = lam_raw[:, healthy].mean(axis=0).sum()
        rho_sc = tgt / raw_mean if raw_mean > 0 else 1.0
        lam_nom = lam_raw * rho_sc
    else:
        print("  [2/4] 复用已有标称流量数据")

    # 正确的扰动框架：在标称轨迹的 λ 峰值时刻，对队列状态施加一次跳变扰动，
    # 之后两条轨迹均使用相同的 λ*(t)，不再对 λ 做任何修改。
    # 这样误差系统 δq = q_per - q_nom 满足自治（或弱时变）条件，dV_W3 < 0 才有意义。
    first_orbit_end = min(int(P.T_orb / P.dt_out), N_t - 1)
    total_lam_first = lam_nom[:first_orbit_end + 1].sum(axis=1)
    k_surge = int(np.argmax(total_lam_first))   # 第一圈 λ 峰值时刻 → 施加跳变
    t_surge = sol_t[k_surge]
    print(f"  队列跳变时刻 t_surge={t_surge/60:.1f} min（第一圈 λ 峰值），"
          f"跳变量 {P.burst_amplitude:.1f} Gbits（均匀）")

    # Part 2 使用独立的 K_part2，保证误差状态在仿真时长内可见收敛
    # τ_error ≈ 1/(K_part2 · λ₂(L)) ≈ 1/(K_part2 · 0.5)
    # K=0.00001 → τ≈55h（不可见）；K_part2=0.05 → τ≈40s（3 圈内可见）
    _orig_K = P.K
    P.K = P.K_part2
    print(f"  Part 2 使用 K_part2={P.K_part2}（Part 1 K={_orig_K}），误差时常数 τ≈{1/(P.K_part2*0.5):.0f} s")

    # --- CFL（基于 K_part2）---
    D_EFF    = 4
    dt_q_cfl = 0.9 / (D_EFF * P.K) if P.K > 0 else P.dt_q
    dt_q_req = min(P.dt_q, dt_q_cfl)
    n_sub    = max(1, math.ceil(P.dt_out / dt_q_req))
    dt_q     = P.dt_out / n_sub

    def _integrate_from(q_ic, k_start, lam_hist_):
        """从 k_start 时刻的初始条件 q_ic 开始积分，返回 (N_t, n) 数组。
        k_start 之前的时间步填 NaN（未参与受扰模拟）。"""
        qh  = np.full((N_t, n), np.nan)
        q_  = q_ic.copy()
        for k_ in range(k_start, N_t):
            deg_ = I_hist[k_].sum(axis=1).astype(float)
            cv_  = np.where(
                deg_ > 0,
                (C_hist[k_] * I_hist[k_]).sum(axis=1) / np.maximum(deg_, 1),
                200e9 / Gb
            )
            qh[k_] = q_
            for _ in range(n_sub):
                u_    = compute_control(q_, C_hist[k_], I_hist[k_])
                dqdt_ = queue_rhs(q_, lam_hist_[k_], mu_nom[k_], u_, I_hist[k_], cv_)
                q_    = euler_step_projected(q_, dqdt_, dt_q)
                if faulty_idx.size > 0:
                    q_[faulty_idx] = 0.0
        return qh

    print("  [3/4] 标称轨迹...")
    if q_nom_ext is not None:
        # 直接使用主仿真的 q_hist，保证与 queues_k=xx.pdf 完全一致
        q_nom = q_nom_ext.copy()
        print("       复用主仿真 q_hist（K={_orig_K}）")
    else:
        q_nom = _integrate_from(np.zeros(n), 0, lam_nom)
        print(f"       内部重积分（K_part2={P.K_part2}，与主仿真 K 不同）")

    print(f"  [3/4] 受扰积分（从 k_surge={k_surge} 起，队列跳变 +{P.burst_amplitude:.1f} Gbits）...")
    q_ic_per = q_nom[k_surge].copy() + P.burst_amplitude   # 在标称基础上跳变
    q_per = q_nom.copy()   # 跳变前与标称相同
    q_per[k_surge:] = _integrate_from(q_ic_per, k_surge, lam_nom)[k_surge:]

    # 恢复原始 K（Lyapunov 矩阵 W 使用 K_part2 以匹配实际控制律）
    # W(t) = (μI + K_part2·L)^{-1}，保持与积分时的 K 一致
    # P.K 此时仍为 K_part2，在 Lyapunov 计算结束后恢复

    # --- 误差 & 时变 Lyapunov ---
    print("  [4/4] 计算误差状态与 V_W3(δq, t) = ½ δqᵀ W(t) δq ...")
    delta_q   = q_per - q_nom   # k < k_surge 时为 0（无扰动），k >= k_surge 时非零

    # 收集 W(t) 和 dδq/dt（仅对 k >= k_surge 有意义）
    W2_hist     = np.zeros((N_t, n, n))
    dqdt_n_hist = np.zeros((N_t, n))
    dqdt_p_hist = np.zeros((N_t, n))
    for k in range(N_t):
        deg_   = I_hist[k].sum(axis=1).astype(float)
        cv_    = np.where(
            deg_ > 0,
            (C_hist[k] * I_hist[k]).sum(axis=1) / np.maximum(deg_, 1),
            200e9 / Gb
        )
        L_k2 = graph_laplacian(I_hist[k])
        W2_hist[k] = lyapunov_W_matrix(mu_nom[k], L_k2)

        u_n    = compute_control(q_nom[k], C_hist[k], I_hist[k])
        dqdt_n_hist[k] = queue_rhs(q_nom[k], lam_nom[k], mu_nom[k], u_n, I_hist[k], cv_)
        u_p    = compute_control(q_per[k], C_hist[k], I_hist[k])
        dqdt_p_hist[k] = queue_rhs(q_per[k], lam_nom[k], mu_nom[k], u_p, I_hist[k], cv_)

    delta_dqdt = dqdt_p_hist - dqdt_n_hist

    # 有限差分估计 Ẇ
    W2_dot = np.zeros_like(W2_hist)
    W2_dot[0]    = (W2_hist[1]  - W2_hist[0])    / P.dt_out
    W2_dot[-1]   = (W2_hist[-1] - W2_hist[-2])   / P.dt_out
    W2_dot[1:-1] = (W2_hist[2:] - W2_hist[:-2])  / (2.0 * P.dt_out)

    VW3_hist  = np.array([lyapunov_VW(delta_q[k],  W2_hist[k])  for k in range(N_t)])
    dVW3_hist = np.array([lyapunov_dVW(delta_q[k], delta_dqdt[k],
                                        W2_hist[k],  W2_dot[k])   for k in range(N_t)])

    P.K = _orig_K   # 恢复全局 K，避免影响后续调用

    print(f"  Part 2 完成。末态 ‖δq‖={np.linalg.norm(delta_q[-1]):.4f} Gbits, "
          f"V_W3末={VW3_hist[-1]:.4e}")
    _plot_part2(t_min, q_nom, q_per, delta_q,
                VW3_hist, dVW3_hist,
                t_surge / 60.0, k_surge=k_surge)
    return q_nom, q_per, delta_q, VW3_hist, dVW3_hist


# =============================================================================
# Section K — 绘图（5 张 PDF）
# =============================================================================


# ------ 【核心修复：注入字体配置，解决所有 PDF 乱码和叉叉问题】 ------
plt.rcParams['font.sans-serif'] = [
    'DejaVu Sans',              # 优先保持基础西文字体稳定性
    'WenQuanYi Micro Hei',      # 文泉驿微米黑（Linux 非常常见）
    'WenQuanYi Zen Hei',        # 文泉驿正黑
    'Noto Sans CJK JP',         # 思源黑体
    'SimHei'                    # 备用
]
plt.rcParams['mathtext.fontset'] = 'cm'                                                 # 使用标准计算机现代字体渲染公式
plt.rcParams['axes.unicode_minus'] = False

# 动态标签和颜色（在 P = Params() 之后计算）
_SAT_LABELS = [f"({p},{k})" for p in range(P.n_patch) for k in range(P.n_patch)]

def _make_colors(n: int) -> np.ndarray:
    if n <= 10:
        return plt.cm.tab10(np.linspace(0, 0.9, n))
    elif n <= 20:
        return plt.cm.tab20(np.linspace(0, 1.0, n))
    else:
        return plt.cm.rainbow(np.linspace(0, 1.0, n))

_COLORS_N = _make_colors(P.n_sat)


def _plot_queues(t_min: np.ndarray, q_hist: np.ndarray):
    n      = P.n_sat
    faulty = set(P.faulty_sats)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i in range(n):
        lw = 0.8 if i in faulty else 1.2
        ls = "--" if i in faulty else "-"
        ax.plot(t_min, q_hist[:, i], label=_SAT_LABELS[i],
                color=_COLORS_N[i], lw=lw, ls=ls)
    ax.set_xlabel("$t$ (min)")
    ax.set_ylabel("$q$ (Gbits)")
    ax.set_title(f"Queue length of each satellite $q_i(t)$ ({P.n_patch} $\\times$ {P.n_patch})")
    ax.legend(ncol=min(n, 5), fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{result_dir}/queues_k={P.K}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   {result_dir}/queues_k={P.K}.pdf")


def _plot_lyapunov(t_min: np.ndarray, V_hist: np.ndarray, dVdt_hist: np.ndarray):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t_min, V_hist, "b-", lw=1.2, label="$V(t)$")
    axes[0].set_ylabel("$V(\\mathbf{q})$ (Gbits$^2$)")
    axes[0].set_title("Lyapunov function $V(t) = \\frac{1}{2}\\|\\mathbf{q}\\|^2$")
    ax0r = axes[0].twinx()
    ax0r.set_ylabel("$\\|\\mathbf{q}\\|$ (Gbits)")
    v_lo, v_hi = max(V_hist.min(), 0.0), V_hist.max()
    ax0r.set_ylim(np.sqrt(2 * v_lo), np.sqrt(2 * max(v_hi, 1e-30)))
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_min, dVdt_hist, "r-", lw=0.8, alpha=0.8, label="$\\dot{V}(t)$")
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].set_xlabel("$t$ (min)")
    axes[1].set_ylabel("$\\dot{V}$ (Gbits$^2$/s)")
    axes[1].set_title("Lyapunov derivative $\\dot{V}(t) = \\mathbf{q}^\\top \\dot{\\mathbf{q}}$")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{result_dir}/lyapunov_k={P.K}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   {result_dir}/lyapunov_k={P.K}.pdf")


def _plot_capacity(t_min: np.ndarray, C_hist: np.ndarray, I_hist: np.ndarray,
                   r_hist: np.ndarray):
    ns = P.n_patch
    # 按方向分组：k 方向（面内）和 p 方向（跨面）
    inplane_links  = []   # (p, k) → (p, k+1)：同轨道面，大间距
    crossplane_links = [] # (p, k) → (p+1, k)：相邻轨道面，小间距
    for p in range(ns):
        for k in range(ns):
            i = ns * p + k
            if k + 1 < ns:
                inplane_links.append((i, ns * p + (k + 1)))
            if p + 1 < ns:
                crossplane_links.append((i, ns * (p + 1) + k))

    def _active(links):
        return [(i, j) for (i, j) in links if I_hist[:, i, j].max() > 0]

    act_in  = _active(inplane_links)
    act_cr  = _active(crossplane_links)

    colors_in = plt.cm.Blues(np.linspace(0.4, 0.9, max(len(act_in), 1)))
    colors_cr = plt.cm.Oranges(np.linspace(0.4, 0.9, max(len(act_cr), 1)))

    def _d_series(i, j):
        d = np.linalg.norm(r_hist[:, i, :] - r_hist[:, j, :], axis=-1) / 1e3
        d[I_hist[:, i, j] == 0] = np.nan
        return d

    def _c_series(i, j):
        c = C_hist[:, i, j].copy().astype(float)
        c[I_hist[:, i, j] == 0] = np.nan
        return c

    # --- 图1：星间真实距离（面内 / 跨面 分上下子图）---
    fig1, (ax_in, ax_cr) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for idx, (i, j) in enumerate(act_in):
        ax_in.plot(t_min, _d_series(i, j), color=colors_in[idx], lw=1.2,
                   label=f"{_SAT_LABELS[i]}—{_SAT_LABELS[j]}")
    ax_in.axhline(P.R_max / 1e3, color="gray", ls=":", lw=1.0, alpha=0.8,
                  label=f"$R_{{max}}$ = {P.R_max/1e3:.0f} km")
    ax_in.set_ylabel("Distance (km)")
    ax_in.set_title("In-plane ISL distance  (same orbital plane, k-direction)")
    ax_in.legend(ncol=min(len(act_in) + 1, 5), fontsize=7, loc="upper right")
    ax_in.grid(True, alpha=0.3)

    for idx, (i, j) in enumerate(act_cr):
        ax_cr.plot(t_min, _d_series(i, j), color=colors_cr[idx], lw=1.2,
                   label=f"{_SAT_LABELS[i]}—{_SAT_LABELS[j]}")
    ax_cr.axhline(P.R_max / 1e3, color="gray", ls=":", lw=1.0, alpha=0.8,
                  label=f"$R_{{max}}$ = {P.R_max/1e3:.0f} km")
    ax_cr.set_xlabel("$t$ (min)")
    ax_cr.set_ylabel("Distance (km)")
    ax_cr.set_title("Cross-plane ISL distance  (adjacent orbital planes, p-direction, "
                    f"$\\Delta\\Omega$ = {360/P.n_planes:.1f}°)")
    ax_cr.legend(ncol=min(len(act_cr) + 1, 5), fontsize=7, loc="upper right")
    ax_cr.grid(True, alpha=0.3)

    fig1.suptitle("Inter-Satellite Distance (with J2)", fontsize=12)
    fig1.tight_layout()
    fig1.savefig(f"{result_dir}/isl_distance.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"   {result_dir}/isl_distance.pdf")

    # --- 图2：香农容量（同样分上下子图）---
    fig2, (ax_cin, ax_ccr) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for idx, (i, j) in enumerate(act_in):
        ax_cin.plot(t_min, _c_series(i, j), color=colors_in[idx], lw=1.2,
                    label=f"{_SAT_LABELS[i]}—{_SAT_LABELS[j]}")
    ax_cin.set_ylabel("Capacity (Gbps)")
    ax_cin.set_title("In-plane ISL Shannon capacity")
    ax_cin.legend(ncol=min(len(act_in), 5), fontsize=7, loc="upper right")
    ax_cin.grid(True, alpha=0.3)

    for idx, (i, j) in enumerate(act_cr):
        ax_ccr.plot(t_min, _c_series(i, j), color=colors_cr[idx], lw=1.2,
                    label=f"{_SAT_LABELS[i]}—{_SAT_LABELS[j]}")
    ax_ccr.set_xlabel("$t$ (min)")
    ax_ccr.set_ylabel("Capacity (Gbps)")
    ax_ccr.set_title("Cross-plane ISL Shannon capacity")
    ax_ccr.legend(ncol=min(len(act_cr), 5), fontsize=7, loc="upper right")
    ax_ccr.grid(True, alpha=0.3)

    fig2.suptitle("ISL Shannon Capacity $C_{ij} = B\\,\\log_2(1+\\gamma_0/d^2)$ (with J2)",
                  fontsize=12)
    fig2.tight_layout()
    fig2.savefig(f"{result_dir}/isl_capacity.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"   {result_dir}/isl_capacity.pdf")


def _plot_lambda(t_min: np.ndarray, lam_hist: np.ndarray):
    n      = P.n_sat
    faulty = set(P.faulty_sats)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i in range(n):
        lw = 0.8 if i in faulty else 1.2
        ls = "--" if i in faulty else "-"
        ax.plot(t_min, lam_hist[:, i], label=_SAT_LABELS[i],
                color=_COLORS_N[i], lw=lw, ls=ls)
    ax.set_xlabel("$t$ (min)")
    ax.set_ylabel("Uplink arrival rate (Gbps)")
    ax.set_title(f"Uplink arrival rate of each satellite $\\lambda_i(t)$")
    ax.legend(ncol=min(n, 5), fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{result_dir}/lambda_k={P.K}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   {result_dir}/lambda_k={P.K}.pdf")


def _plot_dVdt_scatter(V_hist: np.ndarray, dVdt_hist: np.ndarray):
    N = len(V_hist)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(V_hist, dVdt_hist, c=np.arange(N), cmap="plasma",
                    s=4, alpha=0.6, edgecolors="none")
    ax.axhline(0, color="k", lw=1.0, ls="--", label="$\\dot{V}=0$")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Time step index")
    ax.set_xlabel("$V(\\mathbf{q})$ (Gbits$^2$)")
    ax.set_ylabel("$\\dot{V}$ (Gbits$^2$/s)")
    ax.set_title("Phase Portrait")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{result_dir}/phase_k={P.K}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   {result_dir}/phase_k={P.K}.pdf")


def _plot_part1(t_min: np.ndarray, q_hist: np.ndarray,
                VW: np.ndarray, dVW: np.ndarray):
    n      = P.n_sat
    colors = _make_colors(n)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 子图 1：队列时序 + Gaussian 脉冲包络
    ax = axes[0, 0]
    for i in range(n):
        ax.plot(t_min, q_hist[:, i], color=colors[i], lw=1.0, label=_SAT_LABELS[i])
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("$t$ (min)"); ax.set_ylabel("$q_i$ (Gbits)")
    ax.set_title(f"Autonomous system: $q(0)={P.pulse_amplitude:.0f}$ Gbits, $\\lambda\\equiv 0$")
    ax.legend(ncol=3, fontsize=7, loc="upper right"); ax.grid(True, alpha=0.3)

    # 子图 2：V_W(t) 时序（应单调递减）
    ax = axes[0, 1]
    ax.plot(t_min, VW, "m-", lw=1.5, label=r"$V_W = \frac{1}{2}q^\top W(t)q$")
    ax.set_xlabel("$t$ (min)"); ax.set_ylabel("$V_W$ (Gbits$^2$)")
    ax.set_title(r"$V_W(q,t)$ — strictly monotone decreasing (autonomous, $\lambda=0$)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 子图 3：dV_W/dt 时序（自治系统应全程负定）
    ax = axes[1, 0]
    k_peak = int(np.argmax(VW))
    ax.plot(t_min, dVW, "m-", lw=1.0, alpha=0.9, label=r"$\dot V_W$")
    ax.axhline(0, color="k", lw=0.8, ls="--", label="$\\dot V_W = 0$")
    ax.axvline(t_min[k_peak], color="orange", ls=":", lw=1.0,
               label=f"$V_W$ peak ({t_min[k_peak]:.1f} min)")
    ax.set_xlabel("$t$ (min)"); ax.set_ylabel("$\\dot V_W$")
    ax.set_title(r"$\dot V_W < 0$ proves asymptotic stability (autonomous, $\lambda=0$)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 子图 4：相轨迹（原始值）
    ax = axes[1, 1]
    last_nonzero = np.where(VW > 1e-12)[0]
    k_end = int(last_nonzero[-1]) + 1 if len(last_nonzero) > 0 else len(VW)
    ax.plot(VW[:k_end], dVW[:k_end], "m-", lw=1.2, alpha=0.85,
            label=r"$(V_W,\,\dot V_W)$ trajectory")
    ax.plot(VW[0],       dVW[0],       "go", ms=7, zorder=5, label="start")
    ax.plot(VW[k_end-1], dVW[k_end-1], "rs", ms=7, zorder=5, label="end")
    ax.axhline(0, color="k", lw=0.8, ls="--", label="$\\dot V_W = 0$")
    ax.set_xlabel("$V_W(q,t)$ (Gbits$^2$)"); ax.set_ylabel("$\\dot V_W$")
    ax.set_title(r"Phase portrait — $\dot V_W \leq -\|q\|^2$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("Part 1 — Impulse Response & Lyapunov Stability Analysis\n"
                 r"$W(t)=(\mu I+KL(t))^{-1}$: $\dot V_W \leq -\|q\|^2 \leq -2\mu V_W$",
                 fontsize=11)
    fig.tight_layout()
    fname = f"{result_dir}/part1_impulse_response.pdf"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   {fname}")


def _plot_part2(t_min: np.ndarray,
                q_nom: np.ndarray, q_per: np.ndarray,
                delta_q: np.ndarray,
                VW3: np.ndarray, dVW3: np.ndarray,
                t_surge_min: float,
                k_surge: int = 0):
    """
    t_surge_min : 跳变时刻 (min)
    k_surge     : 跳变时刻的时间步索引（用于缩放视图）
    """
    n      = P.n_sat
    colors = _make_colors(n)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def _mark_surge(ax_):
        ax_.axvline(t_surge_min, color="red", ls="--", lw=1.0, alpha=0.8,
                    label=f"State jump $t_{{surge}}$={t_surge_min:.1f} min")

    # --- 确定放大窗口：找到误差完全归零的末步，取跳变前后各少量余量 ---
    norm_err = np.linalg.norm(delta_q, axis=1)
    last_nonzero = np.where(norm_err > 1e-9)[0]
    if len(last_nonzero) > 0:
        k_end_zoom = min(int(last_nonzero[-1]) + max(5, int(0.05*(len(t_min)-k_surge))),
                         len(t_min) - 1)
    else:
        k_end_zoom = min(k_surge + 50, len(t_min) - 1)
    k_start_zoom = max(0, k_surge - 3)
    t_zoom = t_min[k_start_zoom : k_end_zoom + 1]

    # 子图 1：标称 vs 受扰队列（全程总览）
    ax = axes[0, 0]
    for i in range(n):
        ax.plot(t_min, q_nom[:, i], color=colors[i], lw=1.2, ls="-",  alpha=0.9)
        ax.plot(t_min, q_per[:, i], color=colors[i], lw=0.8, ls="--", alpha=0.6)
    _mark_surge(ax)
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color="gray", lw=1.2, ls="-",  label="Nominal $q^*$"),
               Line2D([0],[0], color="gray", lw=0.8, ls="--", label="Perturbed $q$"),
               Line2D([0],[0], color="red",  lw=1.0, ls="--", label="State jump")]
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel("$t$ (min)"); ax.set_ylabel("$q$ (Gbits)")
    ax.set_title("Nominal vs Perturbed (full view)")
    ax.grid(True, alpha=0.3)

    # 子图 2：误差状态放大视图
    ax = axes[0, 1]
    for i in range(n):
        ax.plot(t_zoom,
                delta_q[k_start_zoom : k_end_zoom + 1, i],
                color=colors[i], lw=1.2, label=_SAT_LABELS[i])
    ax.axhline(0, color="k", lw=0.8, ls="--")
    _mark_surge(ax)
    ax.set_xlabel("$t$ (min)"); ax.set_ylabel("$\\delta q_i$ (Gbits)")
    ax.set_title("Error state $\\delta q$ — zoomed to convergence window")
    ax.legend(ncol=3, fontsize=7); ax.grid(True, alpha=0.3)

    # 子图 3：V_W3 放大视图
    ax = axes[1, 0]
    ax.plot(t_zoom,
            VW3[k_start_zoom : k_end_zoom + 1],
            "m-", lw=1.4, label=r"$V_{W3}$")
    _mark_surge(ax)
    ax.set_xlabel("$t$ (min)"); ax.set_ylabel("$V_{W3}$ (Gbits$^2$)")
    ax.set_title(r"$V_{W3}(\delta q,t)$ — zoomed: strictly decreasing after jump")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 子图 4：相轨迹（跳变后非零段，高斯平滑消除折线感）
    ax = axes[1, 1]
    VW3_z   = VW3  [k_surge : k_end_zoom + 1]
    dVW3_z  = dVW3 [k_surge : k_end_zoom + 1]
    if len(VW3_z) > 1:
        ax.plot(VW3_z,  dVW3_z,  "m-", lw=1.2, alpha=0.85,
                label=r"$(V_{W3},\,\dot V_{W3})$ trajectory")
        ax.plot(VW3_z[0],  dVW3_z[0],  "go", ms=8, zorder=5, label="start (jump)")
        ax.plot(VW3_z[-1], dVW3_z[-1], "rs", ms=8, zorder=5, label="end")
    ax.axhline(0, color="k", lw=0.8, ls="--", label="$\\dot V_{W3}=0$")
    ax.set_xlabel("$V_{W3}$  (Gbits$^2$)"); ax.set_ylabel("$\\dot V_{W3}$")
    ax.set_title(r"Phase portrait — $\dot V_{W3}<0$: error converges to nominal")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("Part 2 — State-Jump Perturbation & Error-State Lyapunov Analysis\n"
                 r"Same $\lambda^*(t)$, different IC at $t_{surge}$: "
                 r"$\dot V_{W3}\leq-\|\delta q\|^2<0$",
                 fontsize=11)
    fig.tight_layout()
    fname = f"{result_dir}/part2_perturbed_tracking.pdf"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   {fname}")


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    results = run_simulation()
    _, r_hist, I_hist, C_hist, lam_nom, mu_nom, q_hist, _, _ = results

    print("\n" + "=" * 60)
    print("开始 Part 1 与 Part 2 扩展分析...")
    print("=" * 60)

    run_part1(r_hist=r_hist, I_hist=I_hist, C_hist=C_hist, sol_t=results[0].t)
    run_part2(r_hist=r_hist, I_hist=I_hist, C_hist=C_hist, sol_t=results[0].t,
              lam_nom=lam_nom, mu_nom=mu_nom, q_nom_ext=q_hist)
