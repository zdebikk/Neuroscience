#!/usr/bin/env python3
"""
game1_lif_network.py - Excitatory-Inhibitory Leaky Integrate-and-Fire (LIF) Network
Standalone real-time spiking neural network simulator with Erdos-Renyi connectivity,
vectorized Euler integration, ultra-slow motion & single-step inspection, traveling
synaptic action potential pulses with directional target arrows, and 4-panel
Matplotlib electrophysiological analysis.
"""

import sys
import math
import multiprocessing as mp

# Lock Windows DPI awareness BEFORE importing pygame or matplotlib so opening
# a Matplotlib figure never shrinks or rescales the Pygame window.


import numpy as np
import pygame
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# =============================================================================
# 1. WINDOW & VISUAL THEME CONSTANTS (Adjustable Resolution)
# =============================================================================
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
SIDEBAR_WIDTH = 370
CANVAS_WIDTH = WINDOW_WIDTH - SIDEBAR_WIDTH
FPS = 60

# Minimalist Dark Grey & Magenta Theme
COLOR_BG = (18, 18, 18)             # #121212 Very Dark Grey
COLOR_PANEL = (30, 30, 30)          # #1E1E1E Dark Grey Sidebar
COLOR_CARD = (38, 38, 38)           # #262626 HUD Section Card
COLOR_BORDER = (55, 55, 55)         # #373737 Crisp Divider
COLOR_TEXT = (224, 224, 224)        # #E0E0E0 Crisp Light Grey
COLOR_TEXT_DIM = (145, 145, 145)    # Muted Grey for labels
COLOR_EXC = (255, 0, 127)           # #FF007F Vibrant Magenta (Excitatory)
COLOR_INH = (220, 232, 245)         # Crisp Ice-Silver (Inhibitory contrast)

# Matplotlib Hex Equivalents
HEX_BG = "#121212"
HEX_PANEL = "#1E1E1E"
HEX_TEXT = "#E0E0E0"
HEX_GRID = "#2D2D2D"
HEX_EXC = "#FF007F"
HEX_INH = "#00E5FF"

# =============================================================================
# 2. MATHEMATICAL MODEL PARAMETERS & SPEED LADDER
# =============================================================================
N_TOTAL = 100                       # Total neurons (keep ~100-150 for 60 FPS graph)
FRAC_EXC = 0.8                      # 80% Excitatory, 20% Inhibitory

# Physiological Membrane Constants
V_REST = -65.0                      # Resting potential (mV)
V_RESET = -60.0                     # Post-spike reset potential (mV)
V_TH = -50.0                        # Spike threshold potential (mV)
TAU_E = 20.0                        # Excitatory membrane time constant (ms)
TAU_I = 10.0                        # Inhibitory membrane time constant (ms)
DT_DEFAULT = 0.5                    # Integration time step (ms)
HISTORY_MS = 1000.0                 # Rolling recording window (ms)

R_MEMBRANE = 10.0                   # MOhm
TAU_SYN_FILTER = 4.0                # Synaptic current readout filter time constant (ms)

# 15-tier simulation speed ladder (steps per rendered frame at 60 FPS)
SPEED_LEVELS = [
    0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75,
    1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0
]
DEFAULT_SPEED_INDEX = 8  # Default 2.0 steps/frame (1.0 ms/frame)


class LIFNetworkSimulation:
    """
    Vectorized Leaky Integrate-and-Fire (LIF) network with Erdos-Renyi random
    connectivity and a rolling 1000 ms electrophysiological recording buffer.
    """

    def __init__(self, n_total=N_TOTAL):
        self.N = n_total
        self.N_E = int(round(FRAC_EXC * self.N))
        self.N_I = self.N - self.N_E

        self.is_exc = np.zeros(self.N, dtype=bool)
        self.is_exc[:self.N_E] = True
        self.is_inh = ~self.is_exc

        self.tau = np.where(self.is_exc, TAU_E, TAU_I)

        # Tuned Default Parameters for Asynchronous Irregular (AI) Balanced Regime
        self.p_conn = 0.20          # Erdos-Renyi connection probability p
        self.J_EE = 1.90            # E -> E synaptic weight (mV)
        self.J_EI = 8.50            # I -> E synaptic weight magnitude (mV)
        self.J_IE = 2.40            # E -> I synaptic weight (mV)
        self.J_II = 7.50            # I -> I synaptic weight magnitude (mV)
        self.I_ext_base = 17.5      # Baseline suprathreshold tonic drive (mV)
        self.noise_sigma = 4.2      # Independent Gaussian background fluctuations (mV)
        self.I_click = 38.0         # Left-click continuous stimulation current (mV)

        self.dt = DT_DEFAULT
        self.speed_idx = DEFAULT_SPEED_INDEX
        self.step_accumulator = 0.0
        self.sim_time_ms = 0.0
        self.quiescent_mode = False

        # Adjacency and synaptic weight matrices (post-synaptic i, pre-synaptic j)
        self.C = np.zeros((self.N, self.N), dtype=np.float32)
        self.J = np.zeros((self.N, self.N), dtype=np.float32)
        self.J_exc_only = np.zeros((self.N, self.N), dtype=np.float32)
        self.J_inh_only = np.zeros((self.N, self.N), dtype=np.float32)

        # State vectors
        self.V = np.full(self.N, V_REST, dtype=np.float32)
        self.spikes = np.zeros(self.N, dtype=bool)
        self.prev_spikes = np.zeros(self.N, dtype=bool)
        self.spike_glow = np.zeros(self.N, dtype=np.float32)

        # Post-synaptic impact indicators (tracks which cells just received E or I input)
        self.post_exc_flash = np.zeros(self.N, dtype=np.float32)
        self.post_inh_flash = np.zeros(self.N, dtype=np.float32)

        # Filtered synaptic currents (nA)
        self.I_exc_nA = np.zeros(self.N, dtype=np.float32)
        self.I_inh_nA = np.zeros(self.N, dtype=np.float32)
        self.I_ext_nA = np.zeros(self.N, dtype=np.float32)

        # Rolling history buffers for the last 1000 ms (2000 steps at dt=0.5 ms)
        self.buffer_len = int(round(HISTORY_MS / self.dt))
        self.hist_V = np.full((self.buffer_len, self.N), V_REST, dtype=np.float32)
        self.hist_spikes = np.zeros((self.buffer_len, self.N), dtype=bool)
        self.hist_I_exc = np.zeros((self.buffer_len, self.N), dtype=np.float32)
        self.hist_I_inh = np.zeros((self.buffer_len, self.N), dtype=np.float32)
        self.hist_I_tot = np.zeros((self.buffer_len, self.N), dtype=np.float32)
        self.buf_ptr = 0

        self.rebuild_connectivity()
        self.reset_random_firing()

    @property
    def speed_multiplier(self):
        return SPEED_LEVELS[self.speed_idx]

    def rebuild_connectivity(self):
        """Generates a new Erdos-Renyi adjacency matrix c_ij with probability p."""
        rng = np.random.default_rng()
        self.C = (rng.random((self.N, self.N)) < self.p_conn).astype(np.float32)
        np.fill_diagonal(self.C, 0.0)
        self.update_synaptic_weights()

    def update_synaptic_weights(self):
        """Updates synaptic weight matrix J_ij based on E and I block weights."""
        self.J.fill(0.0)
        E = slice(0, self.N_E)
        I = slice(self.N_E, self.N)

        self.J[E, E] = self.J_EE * self.C[E, E]
        self.J[E, I] = -self.J_EI * self.C[E, I]
        self.J[I, E] = self.J_IE * self.C[I, E]
        self.J[I, I] = -self.J_II * self.C[I, I]

        self.J_exc_only.fill(0.0)
        self.J_inh_only.fill(0.0)
        self.J_exc_only[:, E] = self.J[:, E]
        self.J_inh_only[:, I] = self.J[:, I]

    def reset_random_firing(self):
        """Z Key: Randomizes membrane potentials in [V_rest, V_th] and seeds initial spikes."""
        self.quiescent_mode = False
        self.V = np.random.uniform(V_REST, V_TH, size=self.N).astype(np.float32)
        self.spikes = np.random.random(self.N) < 0.12
        self.prev_spikes.fill(False)
        self.V[self.spikes] = V_RESET
        self.spike_glow = self.spikes.astype(np.float32)
        self.post_exc_flash.fill(0.0)
        self.post_inh_flash.fill(0.0)
        self.I_exc_nA.fill(0.8)
        self.I_inh_nA.fill(-0.8)

    def reset_quiescent(self):
        """X Key: Resets all neurons to V_rest and clears all spike & history buffers."""
        self.quiescent_mode = True
        self.V.fill(V_REST)
        self.spikes.fill(False)
        self.prev_spikes.fill(False)
        self.spike_glow.fill(0.0)
        self.post_exc_flash.fill(0.0)
        self.post_inh_flash.fill(0.0)
        self.I_exc_nA.fill(0.0)
        self.I_inh_nA.fill(0.0)
        self.I_ext_nA.fill(0.0)

        self.hist_V.fill(V_REST)
        self.hist_spikes.fill(False)
        self.hist_I_exc.fill(0.0)
        self.hist_I_inh.fill(0.0)
        self.hist_I_tot.fill(0.0)
        self.buf_ptr = 0

    def step(self, stimulated_idx=None):
        """Advances the LIF network by exactly one Euler integration step dt."""
        self.prev_spikes = self.spikes.copy()
        prev_spikes_f = self.prev_spikes.astype(np.float32)

        # 1. Synaptic voltage jumps from pre-synaptic spikes on the previous step
        dv_exc_syn = self.J_exc_only @ prev_spikes_f
        dv_inh_syn = self.J_inh_only @ prev_spikes_f
        dv_syn_total = dv_exc_syn + dv_inh_syn

        # Track post-synaptic impact flashes for visual target feedback
        self.post_exc_flash *= 0.65
        self.post_inh_flash *= 0.65
        self.post_exc_flash[dv_exc_syn > 0.01] = 1.0
        self.post_inh_flash[dv_inh_syn < -0.01] = 1.0

        # 2. External background drive + stochastic synaptic bombardment
        if self.quiescent_mode:
            I_ext_vec = np.zeros(self.N, dtype=np.float32)
        else:
            noise = np.random.normal(0.0, 1.0, size=self.N).astype(np.float32)
            I_ext_vec = self.I_ext_base + self.noise_sigma * math.sqrt(TAU_E / self.dt) * noise

        # Apply mouse left-click continuous excitation if active
        if stimulated_idx is not None and 0 <= stimulated_idx < self.N:
            I_ext_vec[stimulated_idx] += self.I_click

        # 3. Euler integration step for membrane potential V_i(t)
        leak = -(self.V - V_REST)
        dV = (self.dt / self.tau) * (leak + I_ext_vec) + dv_syn_total
        self.V += dV

        # 4. Update filtered synaptic currents (in nA) for Panel 4 analytical traces
        inst_exc_nA = (self.tau * (dv_exc_syn / self.dt)) / R_MEMBRANE
        inst_inh_nA = (self.tau * (dv_inh_syn / self.dt)) / R_MEMBRANE

        ext_pos_nA = np.maximum(0.0, self.I_ext_base if not self.quiescent_mode else 0.0) / R_MEMBRANE
        if stimulated_idx is not None:
            ext_pos_vec = np.full(self.N, ext_pos_nA, dtype=np.float32)
            ext_pos_vec[stimulated_idx] += self.I_click / R_MEMBRANE
        else:
            ext_pos_vec = ext_pos_nA

        alpha_syn = min(1.0, self.dt / TAU_SYN_FILTER)
        self.I_exc_nA += alpha_syn * ((inst_exc_nA + ext_pos_vec) - self.I_exc_nA)
        self.I_inh_nA += alpha_syn * (inst_inh_nA - self.I_inh_nA)

        residual_ext_nA = (
            leak + (I_ext_vec - (self.I_ext_base if not self.quiescent_mode else 0.0))
        ) / R_MEMBRANE
        self.I_ext_nA += alpha_syn * (residual_ext_nA - self.I_ext_nA)

        I_tot_nA = self.I_exc_nA + self.I_inh_nA + self.I_ext_nA

        # 5. Threshold detection and reset
        self.spikes = self.V >= V_TH
        self.V[self.spikes] = V_RESET
        self.V = np.clip(self.V, -80.0, V_TH)

        # Glow decays per simulation step so ultra-slow motion & single-stepping keep spikes visible
        self.spike_glow *= 0.78
        self.spike_glow[self.spikes] = 1.0

        # 6. Record to circular 1000 ms buffer
        idx = self.buf_ptr
        self.hist_V[idx] = self.V
        self.hist_spikes[idx] = self.spikes
        self.hist_I_exc[idx] = self.I_exc_nA
        self.hist_I_inh[idx] = self.I_inh_nA
        self.hist_I_tot[idx] = I_tot_nA

        self.buf_ptr = (self.buf_ptr + 1) % self.buffer_len
        self.sim_time_ms += self.dt

    def get_ordered_history(self):
        """Returns chronologically ordered arrays covering the last 1000 ms."""
        order = np.roll(np.arange(self.buffer_len), -self.buf_ptr)
        t_axis = np.linspace(0.0, HISTORY_MS, self.buffer_len, endpoint=False)
        return (
            t_axis,
            self.hist_V[order],
            self.hist_spikes[order],
            self.hist_I_exc[order],
            self.hist_I_inh[order],
            self.hist_I_tot[order],
        )


# =============================================================================
# 3. SYNAPTIC PULSE VISUALIZER (TRAVELING PULSES & DIRECTIONAL ARROWS)
# =============================================================================
class SynapticPulseVisualizer:
    """
    Manages animated action potential packets traveling along directed edges
    from pre-synaptic neurons (j) to post-synaptic targets (i), complete with
    directional arrowheads and target impact rings.
    """

    def __init__(self):
        # Vectorized arrays of active pulses: [pre_idx, post_idx, progress (0..1), is_exc]
        self.pre_indices = np.empty(0, dtype=np.int32)
        self.post_indices = np.empty(0, dtype=np.int32)
        self.progress = np.empty(0, dtype=np.float32)
        self.is_exc = np.empty(0, dtype=bool)
        # Keep most recent step's active edges highlighted during paused single-step inspection
        self.step_pre_indices = np.empty(0, dtype=np.int32)
        self.step_post_indices = np.empty(0, dtype=np.int32)
        self.step_is_exc = np.empty(0, dtype=bool)

    def clear(self):
        self.__init__()

    def spawn_from_spikes(self, sim, selected_neuron, max_pulses=450):
        """Spawns traveling pulses along outgoing edges (c_ij = 1) for all newly spiked neurons."""
        spiked_nodes = np.nonzero(sim.spikes)[0]
        if len(spiked_nodes) == 0:
            self.step_pre_indices = np.empty(0, dtype=np.int32)
            self.step_post_indices = np.empty(0, dtype=np.int32)
            self.step_is_exc = np.empty(0, dtype=bool)
            return

        # Prioritize showing all outgoing connections from the selected/clicked neuron first
        if selected_neuron in spiked_nodes:
            spiked_nodes = np.concatenate(
                ([selected_neuron], spiked_nodes[spiked_nodes != selected_neuron])
            )

        # Find all directed edges (post_i, pre_j) where pre_j spiked
        sub_C = sim.C[:, spiked_nodes]
        post_rows, spiked_cols = np.nonzero(sub_C)
        pre_cols = spiked_nodes[spiked_cols]

        if len(pre_cols) == 0:
            return

        # Cap total simultaneous pulses to keep rendering locked at 60 FPS
        if len(pre_cols) > max_pulses:
            pre_cols = pre_cols[:max_pulses]
            post_rows = post_rows[:max_pulses]

        exc_flags = pre_cols < sim.N_E

        # Save the latest step's active edges for persistent inspection while paused
        self.step_pre_indices = pre_cols.copy()
        self.step_post_indices = post_rows.copy()
        self.step_is_exc = exc_flags.copy()

        # Append to traveling pulse arrays
        self.pre_indices = np.concatenate([self.pre_indices, pre_cols])
        self.post_indices = np.concatenate([self.post_indices, post_rows])
        self.progress = np.concatenate(
            [self.progress, np.zeros(len(pre_cols), dtype=np.float32)]
        )
        self.is_exc = np.concatenate([self.is_exc, exc_flags])

        # Enforce global cap on in-flight pulses
        if len(self.pre_indices) > max_pulses:
            self.pre_indices = self.pre_indices[-max_pulses:]
            self.post_indices = self.post_indices[-max_pulses:]
            self.progress = self.progress[-max_pulses:]
            self.is_exc = self.is_exc[-max_pulses:]

    def update(self, frame_dt, paused, speed_mult):
        """Advances traveling pulses along edges in real wall-clock time."""
        if len(self.progress) == 0:
            return

        # When paused (e.g. stepping with RIGHT ARROW) or in ultra-slow motion,
        # glide pulses smoothly across the axon over ~0.55 seconds so the user
        # can clearly see the direction of propagation and target impact.
        if paused or speed_mult <= 0.35:
            transit_seconds = 0.55
        elif speed_mult <= 1.0:
            transit_seconds = 0.32
        else:
            transit_seconds = 0.18

        self.progress += frame_dt / transit_seconds

        # Keep pulses slightly past 1.0 (up to 1.15) to render a brief target arrival flash
        keep = self.progress <= 1.15
        self.pre_indices = self.pre_indices[keep]
        self.post_indices = self.post_indices[keep]
        self.progress = self.progress[keep]
        self.is_exc = self.is_exc[keep]

    @staticmethod
    def _draw_arrowhead(surface, color, x1, y1, x2, y2, tip_offset=10.0, size=7.0):
        """Draws a directional arrowhead along (x1, y1) -> (x2, y2) just before the target node."""
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        if dist < 24.0:
            return
        ux = dx / dist
        uy = dy / dist

        # Place arrow tip slightly outside the post-synaptic node radius
        tx = x2 - ux * tip_offset
        ty = y2 - uy * tip_offset

        # Perpendicular normal vector
        px = -uy
        py = ux

        bx = tx - ux * size
        by = ty - uy * size

        left_pt = (int(bx + px * (size * 0.52)), int(by + py * (size * 0.52)))
        right_pt = (int(bx - px * (size * 0.52)), int(by - py * (size * 0.52)))
        tip_pt = (int(tx), int(ty))

        pygame.draw.polygon(surface, color, [tip_pt, left_pt, right_pt])

    def draw(self, surface, positions, paused, speed_mult, selected_neuron):
        """Renders highlighted synaptic pathways, arrowheads, and traveling pulse packets."""
        show_detailed_arrows = paused or (speed_mult <= 0.50)

        # 1. If paused (e.g., inspecting via RIGHT ARROW), keep the current step's
        # active outgoing synapses and target arrowheads illuminated.
        if paused and len(self.step_pre_indices) > 0:
            for idx in range(len(self.step_pre_indices)):
                pre_i = self.step_pre_indices[idx]
                post_i = self.step_post_indices[idx]
                exc = self.step_is_exc[idx]
                is_sel = (pre_i == selected_neuron) or (post_i == selected_neuron)

                x1, y1 = positions[pre_i]
                x2, y2 = positions[post_i]

                if exc:
                    line_col = (220, 0, 110) if is_sel else (135, 0, 68)
                    arrow_col = (255, 50, 150) if is_sel else (195, 0, 98)
                else:
                    line_col = (160, 210, 240) if is_sel else (95, 125, 150)
                    arrow_col = (210, 240, 255) if is_sel else (150, 185, 215)

                pygame.draw.aaline(surface, line_col, (int(x1), int(y1)), (int(x2), int(y2)))
                self._draw_arrowhead(surface, arrow_col, x1, y1, x2, y2, tip_offset=11.0, size=7.5)

        # 2. Draw all actively traveling action potential pulses
        for idx in range(len(self.pre_indices)):
            pre_i = self.pre_indices[idx]
            post_i = self.post_indices[idx]
            prog = float(self.progress[idx])
            exc = bool(self.is_exc[idx])
            is_sel = (pre_i == selected_neuron) or (post_i == selected_neuron)

            x1, y1 = positions[pre_i]
            x2, y2 = positions[post_i]

            clamped_p = min(1.0, prog)
            fade = 1.0 - max(0.0, (prog - 0.85) / 0.30)

            if exc:
                line_col = (int(190 * fade), 0, int(95 * fade))
                pulse_col = (255, 90, 175) if is_sel else (255, 20, 135)
                tail_col = (180, 0, 90)
            else:
                line_col = (int(130 * fade), int(165 * fade), int(195 * fade))
                pulse_col = (220, 245, 255)
                tail_col = (120, 165, 200)

            # Underlying illuminated axon line
            if not paused:
                pygame.draw.aaline(surface, line_col, (int(x1), int(y1)), (int(x2), int(y2)))

            # Directional arrowhead at target when in slow-motion or connected to selected cell
            if (show_detailed_arrows or is_sel) and not paused:
                self._draw_arrowhead(surface, pulse_col, x1, y1, x2, y2, tip_offset=11.0, size=7.0)

            # Traveling action potential comet (head + short trailing wake)
            if prog <= 1.0:
                px = x1 + (x2 - x1) * clamped_p
                py = y1 + (y2 - y1) * clamped_p
                tail_p = max(0.0, clamped_p - 0.14)
                tx = x1 + (x2 - x1) * tail_p
                ty = y1 + (y2 - y1) * tail_p

                pygame.draw.line(surface, tail_col, (int(tx), int(ty)), (int(px), int(py)), 2)
                pygame.draw.circle(surface, pulse_col, (int(px), int(py)), 4 if is_sel else 3)
            else:
                # Pulse just arrived at post-synaptic target -> draw expanding arrival ring
                ring_r = int(8 + (prog - 1.0) * 45.0)
                pygame.draw.circle(surface, pulse_col, (int(x2), int(y2)), ring_r, width=1)


# =============================================================================
# 4. GRAPH LAYOUT & EDGE SURFACE CACHE
# =============================================================================
def compute_node_positions(n_total, n_exc, width, height):
    """Arranges Inhibitory neurons in an inner ring and Excitatory in two outer rings."""
    cx, cy = width * 0.5, height * 0.52
    positions = np.zeros((n_total, 2), dtype=np.float32)

    n_inner_e = n_exc // 2
    n_outer_e = n_exc - n_inner_e

    for i in range(n_inner_e):
        angle = 2.0 * math.pi * i / n_inner_e + 0.15
        r = 195.0 + 14.0 * math.sin(i * 2.4)
        positions[i] = [cx + r * math.cos(angle), cy + r * math.sin(angle)]

    for i in range(n_outer_e):
        idx = n_inner_e + i
        angle = 2.0 * math.pi * i / n_outer_e
        r = 305.0 + 16.0 * math.cos(i * 3.1)
        positions[idx] = [cx + r * math.cos(angle), cy + r * math.sin(angle)]

    n_inh = n_total - n_exc
    for i in range(n_inh):
        idx = n_exc + i
        angle = 2.0 * math.pi * i / max(1, n_inh)
        r = 95.0 + 10.0 * math.sin(i * 1.7)
        positions[idx] = [cx + r * math.cos(angle), cy + r * math.sin(angle)]

    return positions


def build_edge_surface(C, positions, width, height):
    """Pre-renders active synaptic connections (c_ij = 1) onto a translucent surface."""
    edge_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    rows, cols = np.nonzero(C)
    line_color = (110, 110, 120, 24)
    for r, c in zip(rows, cols):
        p1 = (int(positions[c, 0]), int(positions[c, 1]))
        p2 = (int(positions[r, 0]), int(positions[r, 1]))
        pygame.draw.line(edge_surf, line_color, p1, p2, 1)
    return edge_surf

#WINDOW SCALING NOTE: Matplotlib figures must be generated in a separate process to avoid

def _plot_worker(plot_data):
    """Runs inside a separate process so Matplotlib never alters Pygame's Windows DPI scaling."""
    t_axis = plot_data["t_axis"]
    V_hist = plot_data["V_hist"]
    spikes_hist = plot_data["spikes_hist"]
    I_exc_hist = plot_data["I_exc_hist"]
    I_inh_hist = plot_data["I_inh_hist"]
    I_tot_hist = plot_data["I_tot_hist"]
    selected_neuron = plot_data["selected_neuron"]
    N = plot_data["N"]
    N_E = plot_data["N_E"]
    dt = plot_data["dt"]
    p_conn = plot_data["p_conn"]
    J_EE, J_EI, J_IE, J_II = plot_data["J_weights"]

# =============================================================================
# 5. 4-PANEL MATPLOTLIB ANALYTICAL FIGURE (P Key)
# =============================================================================
def show_analytical_plots(sim, selected_neuron):
    """
    Generates and displays the 4-panel Matplotlib electrophysiological summary
    for the last 1000 ms of simulation history, styled in Dark Grey & Magenta.
    """
    t_axis, V_hist, spikes_hist, I_exc_hist, I_inh_hist, I_tot_hist = sim.get_ordered_history()

    plt.rcParams.update({
        "figure.facecolor": HEX_BG,
        "axes.facecolor": HEX_PANEL,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": HEX_TEXT,
        "xtick.color": HEX_TEXT,
        "ytick.color": HEX_TEXT,
        "text.color": HEX_TEXT,
        "font.family": "monospace",
        "font.size": 9,
    })

    fig, axes = plt.subplots(
        4, 1, figsize=(11, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0, 1.15, 1.25], "hspace": 0.22}
    )
    fig.canvas.manager.set_window_title(
        f"Electrophysiological Analysis (Last 1000 ms) - Recorded Neuron #{selected_neuron + 1}"
    )

    # PANEL 1: Spike Times (Raster Plot)
    ax1 = axes[0]
    step_indices, neuron_indices = np.nonzero(spikes_hist)
    spike_times = t_axis[step_indices]
    cell_numbers = neuron_indices + 1

    exc_mask = neuron_indices < sim.N_E
    inh_mask = ~exc_mask

    ax1.scatter(
        spike_times[exc_mask], cell_numbers[exc_mask],
        s=6, c=HEX_EXC, marker="|", linewidths=1.1, label=f"Excitatory (1-{sim.N_E})"
    )
    ax1.scatter(
        spike_times[inh_mask], cell_numbers[inh_mask],
        s=8, c=HEX_INH, marker="|", linewidths=1.2, label=f"Inhibitory ({sim.N_E + 1}-{sim.N})"
    )
    ax1.axhline(selected_neuron + 1, color="#FFFFFF", linestyle=":", linewidth=0.8, alpha=0.6)
    ax1.set_ylabel("Cell Number")
    ax1.set_ylim(0, sim.N + 2)
    ax1.set_title(
        f"1. Spike Raster Plot & Network Activity (p={sim.p_conn:.2f}, "
        f"J_EE={sim.J_EE:.1f}, J_EI={sim.J_EI:.1f}, J_IE={sim.J_IE:.1f}, J_II={sim.J_II:.1f})",
        loc="left", color=HEX_TEXT, fontweight="bold", pad=6
    )
    ax1.legend(loc="upper right", facecolor=HEX_BG, edgecolor="#444444", framealpha=0.9)
    ax1.grid(True, color=HEX_GRID, linestyle="--", linewidth=0.5, alpha=0.5)

    # PANEL 2: Population Activity (Hz)
    ax2 = axes[1]
    bin_ms = 2.0
    steps_per_bin = max(1, int(round(bin_ms / sim.dt)))
    num_bins = len(t_axis) // steps_per_bin

    trimmed_spikes = spikes_hist[:num_bins * steps_per_bin]
    binned_counts = trimmed_spikes.reshape(num_bins, steps_per_bin, sim.N).sum(axis=(1, 2))
    pop_rate_hz = binned_counts / (sim.N * (steps_per_bin * sim.dt / 1000.0))
    bin_centers = np.linspace(bin_ms * 0.5, HISTORY_MS - bin_ms * 0.5, num_bins)

    kernel_half = 9
    kx = np.arange(-kernel_half, kernel_half + 1) * bin_ms
    kernel = np.exp(-0.5 * (kx / 6.0) ** 2)
    kernel /= kernel.sum()
    smooth_rate_hz = np.convolve(pop_rate_hz, kernel, mode="same")

    ax2.bar(
        bin_centers, pop_rate_hz, width=bin_ms,
        color="#484848", alpha=0.65, label=f"Raw Binned Rate ({bin_ms:.0f} ms)"
    )
    ax2.plot(
        bin_centers, smooth_rate_hz,
        color=HEX_EXC, linewidth=1.8, label="Smoothed Population Rate (Hz)"
    )
    ax2.set_ylabel("Pop. Rate (Hz)")
    ax2.set_title("2. Instantaneous Population Firing Rate", loc="left", fontweight="bold", pad=6)
    ax2.legend(loc="upper right", facecolor=HEX_BG, edgecolor="#444444", framealpha=0.9)
    ax2.grid(True, color=HEX_GRID, linestyle="--", linewidth=0.5, alpha=0.5)

    # PANEL 3: Membrane Potential (mV) of Selected Neuron
    ax3 = axes[2]
    v_trace = V_hist[:, selected_neuron].copy()
    cell_spikes = spikes_hist[:, selected_neuron]
    n_type_str = "Excitatory" if selected_neuron < sim.N_E else "Inhibitory"
    trace_color = HEX_EXC if selected_neuron < sim.N_E else HEX_INH

    ax3.plot(t_axis, v_trace, color=trace_color, linewidth=1.2, label=f"V_m(t) Cell #{selected_neuron + 1}")
    spike_t = t_axis[cell_spikes]
    if len(spike_t) > 0:
        ax3.vlines(
            spike_t, V_TH, 0.0,
            colors="#FFFFFF", linewidth=1.1, label=f"Action Potentials ({len(spike_t)} spikes)"
        )

    ax3.axhline(V_TH, color="#FF5555", linestyle="--", linewidth=0.9, label="V_th (-50 mV)")
    ax3.axhline(V_RESET, color="#777777", linestyle=":", linewidth=0.8, label="V_reset (-60 mV)")
    ax3.set_ylabel("Voltage (mV)")
    ax3.set_ylim(-68.0, 5.0)
    ax3.set_title(
        f"3. Intracellular Membrane Potential - Neuron #{selected_neuron + 1} ({n_type_str})",
        loc="left", fontweight="bold", pad=6
    )
    ax3.legend(loc="upper right", facecolor=HEX_BG, edgecolor="#444444", framealpha=0.9, ncol=2)
    ax3.grid(True, color=HEX_GRID, linestyle="--", linewidth=0.5, alpha=0.5)

    # PANEL 4: Membrane Currents (nA) of Selected Neuron
    ax4 = axes[3]
    i_exc = I_exc_hist[:, selected_neuron]
    i_inh = I_inh_hist[:, selected_neuron]
    i_tot = I_tot_hist[:, selected_neuron]

    ax4.plot(t_axis, i_exc, color=HEX_EXC, linewidth=1.3, label="I_exc > 0 (Excitatory)")
    ax4.plot(t_axis, i_inh, color=HEX_INH, linewidth=1.3, label="I_inh < 0 (Inhibitory)")
    ax4.plot(t_axis, i_tot, color=HEX_TEXT, linewidth=1.1, alpha=0.9, label="I_tot = I_exc + I_inh + I_ext")
    ax4.axhline(0.0, color="#888888", linestyle="--", linewidth=0.9)

    ax4.set_xlabel("Simulated Time Window (ms)")
    ax4.set_ylabel("Current (nA)")
    ax4.set_xlim(0.0, HISTORY_MS)
    ax4.set_title(
        f"4. Synaptic & Net Membrane Currents - Neuron #{selected_neuron + 1} (E-I Balance)",
        loc="left", fontweight="bold", pad=6
    )
    ax4.legend(loc="upper right", facecolor=HEX_BG, edgecolor="#444444", framealpha=0.9, ncol=3)
    ax4.grid(True, color=HEX_GRID, linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    plt.show()
    plt.close(fig)


# =============================================================================
# 6. MAIN PYGAME APPLICATION LOOP
# =============================================================================
def main():
    pygame.init()
    pygame.display.set_caption("GAME 1 // E-I LEAKY INTEGRATE-AND-FIRE (LIF) NETWORK")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    font_names = ["consolas", "menlo", "dejavusansmono", "couriernew", "monospace"]
    fonts = {
        "title": pygame.font.SysFont(font_names, 18, bold=True),
        "section": pygame.font.SysFont(font_names, 13, bold=True),
        "hud": pygame.font.SysFont(font_names, 13, bold=False),
        "small": pygame.font.SysFont(font_names, 11, bold=False),
    }

    sim = LIFNetworkSimulation(n_total=N_TOTAL)
    positions = compute_node_positions(sim.N, sim.N_E, CANVAS_WIDTH, WINDOW_HEIGHT)
    edge_surface = build_edge_surface(sim.C, positions, CANVAS_WIDTH, WINDOW_HEIGHT)
    pulse_viz = SynapticPulseVisualizer()
    pulse_viz.spawn_from_spikes(sim, selected_neuron=0)

    selected_neuron = 0
    mouse_stimulating = False
    paused = False

    running = True
    while running:
        frame_dt = clock.tick(FPS) / 1000.0

        # ---------------------------------------------------------------------
        # EVENT HANDLING & KEYBOARD SHORTCUTS
        # ---------------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if mx < CANVAS_WIDTH:
                    dists = np.hypot(positions[:, 0] - mx, positions[:, 1] - my)
                    nearest_idx = int(np.argmin(dists))
                    if dists[nearest_idx] <= 24.0:
                        selected_neuron = nearest_idx
                        mouse_stimulating = True

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mouse_stimulating = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                # SPACE: Pause / Resume
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    sim.step_accumulator = 0.0

                # RIGHT ARROW: Pause and step forward by 1 integration step (dt = 0.5 ms)
                # Spawns traveling pulses & directional arrows for all neurons that fired
                elif event.key == pygame.K_RIGHT:
                    paused = True
                    sim.step_accumulator = 0.0
                    stim_target = selected_neuron if mouse_stimulating else None
                    sim.step(stimulated_idx=stim_target)
                    pulse_viz.spawn_from_spikes(sim, selected_neuron)

                # Q / A: Connectivity probability p (rebuilds c_ij matrix)
                elif event.key == pygame.K_q:
                    sim.p_conn = min(1.0, round(sim.p_conn + 0.02, 2))
                    sim.rebuild_connectivity()
                    edge_surface = build_edge_surface(sim.C, positions, CANVAS_WIDTH, WINDOW_HEIGHT)
                    pulse_viz.clear()
                elif event.key == pygame.K_a:
                    sim.p_conn = max(0.02, round(sim.p_conn - 0.02, 2))
                    sim.rebuild_connectivity()
                    edge_surface = build_edge_surface(sim.C, positions, CANVAS_WIDTH, WINDOW_HEIGHT)
                    pulse_viz.clear()

                # W / S: J_EE (E-to-E weight)
                elif event.key == pygame.K_w:
                    sim.J_EE = round(sim.J_EE + 0.1, 2)
                    sim.update_synaptic_weights()
                elif event.key == pygame.K_s:
                    sim.J_EE = max(0.0, round(sim.J_EE - 0.1, 2))
                    sim.update_synaptic_weights()

                # E / D: J_EI (I-to-E weight)
                elif event.key == pygame.K_e:
                    sim.J_EI = round(sim.J_EI + 0.2, 2)
                    sim.update_synaptic_weights()
                elif event.key == pygame.K_d:
                    sim.J_EI = max(0.0, round(sim.J_EI - 0.2, 2))
                    sim.update_synaptic_weights()

                # R / F: J_IE (E-to-I weight)
                elif event.key == pygame.K_r:
                    sim.J_IE = round(sim.J_IE + 0.1, 2)
                    sim.update_synaptic_weights()
                elif event.key == pygame.K_f:
                    sim.J_IE = max(0.0, round(sim.J_IE - 0.1, 2))
                    sim.update_synaptic_weights()

                # T / G: J_II (I-to-I weight)
                elif event.key == pygame.K_t:
                    sim.J_II = round(sim.J_II + 0.2, 2)
                    sim.update_synaptic_weights()
                elif event.key == pygame.K_g:
                    sim.J_II = max(0.0, round(sim.J_II - 0.2, 2))
                    sim.update_synaptic_weights()

                # Y / H: Baseline external drive I_ext
                elif event.key == pygame.K_y:
                    sim.I_ext_base = round(sim.I_ext_base + 0.5, 2)
                    sim.quiescent_mode = False
                elif event.key == pygame.K_h:
                    sim.I_ext_base = max(0.0, round(sim.I_ext_base - 0.5, 2))

                # Z: Random firing state reset
                elif event.key == pygame.K_z:
                    sim.reset_random_firing()
                    pulse_viz.clear()
                    pulse_viz.spawn_from_spikes(sim, selected_neuron)

                # X: Quiescent reset
                elif event.key == pygame.K_x:
                    sim.reset_quiescent()
                    pulse_viz.clear()

                # UP / DOWN: Adjust speed across fractional slow-mo & fast-forward ladder
                elif event.key == pygame.K_UP:
                    sim.speed_idx = min(len(SPEED_LEVELS) - 1, sim.speed_idx + 1)
                elif event.key == pygame.K_DOWN:
                    sim.speed_idx = max(0, sim.speed_idx - 1)

                # P: Pause and open 4-panel Matplotlib analytical window
                elif event.key == pygame.K_p:
                    was_paused = paused
                    paused = True
                    mouse_stimulating = False
                    show_analytical_plots(sim, selected_neuron)
                    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
                    pygame.event.clear()
                    paused = was_paused

        # If holding left-click while moving near another node, allow smooth selection
        if mouse_stimulating:
            mx, my = pygame.mouse.get_pos()
            if mx < CANVAS_WIDTH:
                dists = np.hypot(positions[:, 0] - mx, positions[:, 1] - my)
                nearest_idx = int(np.argmin(dists))
                if dists[nearest_idx] <= 24.0:
                    selected_neuron = nearest_idx

        # ---------------------------------------------------------------------
        # SIMULATION STEPPING & PULSE SPAWNING
        # ---------------------------------------------------------------------
        if not paused:
            stim_target = selected_neuron if mouse_stimulating else None
            sim.step_accumulator += sim.speed_multiplier
            steps_to_run = int(sim.step_accumulator)
            if steps_to_run > 0:
                sim.step_accumulator -= steps_to_run
                for _ in range(steps_to_run):
                    sim.step(stimulated_idx=stim_target)
                    # Spawn traveling pulses when running at normal or slow speeds
                    if sim.speed_multiplier <= 4.0:
                        pulse_viz.spawn_from_spikes(sim, selected_neuron)

        # Always update traveling pulse positions at 60 FPS (even when paused after RIGHT ARROW)
        pulse_viz.update(frame_dt, paused, sim.speed_multiplier)

        # ---------------------------------------------------------------------
        # RENDERING: NETWORK GRAPH CANVAS
        # ---------------------------------------------------------------------
        screen.fill(COLOR_BG)

        # 1. Blit pre-rendered static Erdos-Renyi connectivity matrix
        screen.blit(edge_surface, (0, 0))

        # 2. Draw traveling synaptic pulses and directional arrowheads
        pulse_viz.draw(screen, positions, paused, sim.speed_multiplier, selected_neuron)

        # 3. Draw Excitatory (Filled Circles) and Inhibitory (Triangles + Rings) Nodes
        v_norm = np.clip((sim.V - V_REST) / (V_TH - V_REST), 0.0, 1.0)

        for i in range(sim.N):
            x, y = int(positions[i, 0]), int(positions[i, 1])
            vn = float(v_norm[i])
            glow = float(sim.spike_glow[i])
            exc_hit = float(sim.post_exc_flash[i])
            inh_hit = float(sim.post_inh_flash[i])

            base_radius = 5.0 + 4.5 * vn + 5.5 * glow

            # Subtle post-synaptic impact halo when a neuron receives synaptic input
            if (paused or sim.speed_multiplier <= 1.0) and max(exc_hit, inh_hit) > 0.35:
                halo_col = (150, 0, 75) if exc_hit >= inh_hit else (90, 145, 180)
                pygame.draw.circle(screen, halo_col, (x, y), int(base_radius + 4), width=1)

            if i < sim.N_E:
                r_col = int(min(255, 65 + 190 * vn + 100 * glow))
                g_col = int(min(255, 190 * glow))
                b_col = int(min(255, 35 + 92 * vn + 140 * glow))
                if glow > 0.25:
                    pygame.draw.circle(
                        screen, (140, 0, 70), (x, y), int(base_radius + 6), width=1
                    )
                pygame.draw.circle(screen, (r_col, g_col, b_col), (x, y), int(base_radius))
            else:
                intensity = 0.35 + 0.65 * vn
                col = (
                    int(min(255, COLOR_INH[0] * intensity + 80 * glow)),
                    int(min(255, COLOR_INH[1] * intensity + 80 * glow)),
                    int(min(255, COLOR_INH[2] * intensity + 80 * glow)),
                )
                r_tri = base_radius + 1.5
                pts = [
                    (x, int(y - r_tri)),
                    (int(x - r_tri * 0.88), int(y + r_tri * 0.65)),
                    (int(x + r_tri * 0.88), int(y + r_tri * 0.65)),
                ]
                pygame.draw.polygon(screen, col, pts, width=0 if glow > 0.4 else 2)
                pygame.draw.circle(screen, col, (x, y), int(base_radius + 3), width=1)

        # 4. Draw Reticle around Selected / Recorded Neuron
        sx, sy = int(positions[selected_neuron, 0]), int(positions[selected_neuron, 1])
        reticle_color = COLOR_EXC if mouse_stimulating else COLOR_TEXT
        pygame.draw.circle(screen, reticle_color, (sx, sy), 16, width=2)
        pygame.draw.line(screen, reticle_color, (sx - 21, sy), (sx - 13, sy), 2)
        pygame.draw.line(screen, reticle_color, (sx + 13, sy), (sx + 21, sy), 2)
        pygame.draw.line(screen, reticle_color, (sx, sy - 21), (sx, sy - 13), 2)
        pygame.draw.line(screen, reticle_color, (sx, sy + 13), (sx, sy + 21), 2)

        # Top-left canvas overlay status
        num_spiked = int(sim.spikes.sum())
        num_targets = len(pulse_viz.step_post_indices)
        state_badge = "PAUSED [RIGHT ARROW = +1 STEP]" if paused else (
            "QUIESCENT" if sim.quiescent_mode else "AI BALANCED ACTIVE"
        )
        badge_col = (255, 175, 0) if paused else (
            COLOR_TEXT_DIM if sim.quiescent_mode else COLOR_EXC
        )
        overlay_surf = fonts["section"].render(
            f"STATE: {state_badge}  |  SIM TIME: {sim.sim_time_ms:8.1f} ms  |  "
            f"SPIKES: {num_spiked} (-> {num_targets} targets)",
            True,
            badge_col,
        )
        screen.blit(overlay_surf, (24, 18))

        if mouse_stimulating:
            stim_surf = fonts["section"].render(
                f">> INJECTING +I_CLICK ({sim.I_click:.0f} mV) INTO NEURON #{selected_neuron + 1} <<",
                True,
                COLOR_EXC,
            )
            screen.blit(stim_surf, (24, 40))

        # ---------------------------------------------------------------------
        # RENDERING: RIGHT SIDEBAR HUD
        # ---------------------------------------------------------------------
        sidebar_rect = pygame.Rect(CANVAS_WIDTH, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(screen, COLOR_PANEL, sidebar_rect)
        pygame.draw.line(screen, COLOR_EXC, (CANVAS_WIDTH, 0), (CANVAS_WIDTH, WINDOW_HEIGHT), 2)

        x_pad = CANVAS_WIDTH + 20
        y_cursor = 20

        t_surf = fonts["title"].render("E-I LIF NETWORK HUD", True, COLOR_TEXT)
        screen.blit(t_surf, (x_pad, y_cursor))
        y_cursor += 24
        sub_surf = fonts["small"].render(
            f"N={sim.N} ({sim.N_E} Exc Magenta Circles, {sim.N_I} Inh Triangles)",
            True,
            COLOR_TEXT_DIM,
        )
        screen.blit(sub_surf, (x_pad, y_cursor))
        y_cursor += 26

        # Section 1: Network & Synaptic Parameters
        card1 = pygame.Rect(x_pad - 6, y_cursor, SIDEBAR_WIDTH - 28, 222)
        pygame.draw.rect(screen, COLOR_CARD, card1, border_radius=5)
        pygame.draw.rect(screen, COLOR_BORDER, card1, width=1, border_radius=5)
        y_cursor += 10

        sec1 = fonts["section"].render("SYNAPTIC & DRIVE PARAMETERS", True, COLOR_EXC)
        screen.blit(sec1, (x_pad + 4, y_cursor))
        y_cursor += 22

        spd = sim.speed_multiplier
        spd_str = (
            f"{spd:.2f}x ({spd * sim.dt * FPS:.1f} ms/s)"
            if spd < 1.0
            else f"{spd:.1f}x ({spd * sim.dt:.1f} ms/f)"
        )

        param_rows = [
            ("[Q/A] Conn Prob (p)", f"{sim.p_conn:.2f} ({int(sim.C.sum())} synapses)"),
            ("[W/S] Weight J_EE (E->E)", f"+{sim.J_EE:.2f} mV"),
            ("[E/D] Weight J_EI (I->E)", f"-{sim.J_EI:.2f} mV"),
            ("[R/F] Weight J_IE (E->I)", f"+{sim.J_IE:.2f} mV"),
            ("[T/G] Weight J_II (I->I)", f"-{sim.J_II:.2f} mV"),
            ("[Y/H] Ext Drive (I_ext)", f"{sim.I_ext_base:.2f} mV"),
            ("[UP/DN] Sim Speed", spd_str),
        ]
        for label, val in param_rows:
            l_surf = fonts["hud"].render(label, True, COLOR_TEXT)
            v_surf = fonts["hud"].render(
                val, True, COLOR_EXC if "Weight" in label or "Speed" in label else COLOR_TEXT
            )
            screen.blit(l_surf, (x_pad + 4, y_cursor))
            screen.blit(v_surf, (CANVAS_WIDTH + SIDEBAR_WIDTH - 18 - v_surf.get_width(), y_cursor))
            y_cursor += 25

        y_cursor += 16

        # Section 2: Live Single-Cell Oscilloscope Preview
        card2 = pygame.Rect(x_pad - 6, y_cursor, SIDEBAR_WIDTH - 28, 175)
        pygame.draw.rect(screen, COLOR_CARD, card2, border_radius=5)
        pygame.draw.rect(screen, COLOR_BORDER, card2, width=1, border_radius=5)
        y_cursor += 10

        cell_type = "EXCITATORY" if selected_neuron < sim.N_E else "INHIBITORY"
        sec2 = fonts["section"].render(
            f"LIVE TRACE: NEURON #{selected_neuron + 1} [{cell_type}]", True, COLOR_EXC
        )
        screen.blit(sec2, (x_pad + 4, y_cursor))
        y_cursor += 20

        v_now = sim.V[selected_neuron]
        i_e_now = sim.I_exc_nA[selected_neuron]
        i_i_now = sim.I_inh_nA[selected_neuron]
        readout = fonts["small"].render(
            f"V_m: {v_now:6.1f} mV  |  I_exc: +{i_e_now:4.2f} nA  |  I_inh: {i_i_now:5.2f} nA",
            True,
            COLOR_TEXT,
        )
        screen.blit(readout, (x_pad + 4, y_cursor))
        y_cursor += 20

        scope_rect = pygame.Rect(x_pad + 4, y_cursor, SIDEBAR_WIDTH - 48, 110)
        pygame.draw.rect(screen, COLOR_BG, scope_rect)
        pygame.draw.rect(screen, COLOR_BORDER, scope_rect, width=1)

        order = np.roll(np.arange(sim.buffer_len), -sim.buf_ptr)[-320:]
        v_recent = sim.hist_V[order, selected_neuron]
        spk_recent = sim.hist_spikes[order, selected_neuron]

        pts = []
        for idx_pt, val in enumerate(v_recent):
            px = scope_rect.x + int((idx_pt / 319.0) * scope_rect.width)
            norm_v = (val - (-70.0)) / (0.0 - (-70.0))
            py = scope_rect.bottom - int(np.clip(norm_v, 0.05, 0.95) * scope_rect.height)
            pts.append((px, py))
            if spk_recent[idx_pt]:
                pygame.draw.line(screen, COLOR_TEXT, (px, py), (px, scope_rect.y + 6), 1)

        if len(pts) > 1:
            trace_col = COLOR_EXC if selected_neuron < sim.N_E else COLOR_INH
            pygame.draw.lines(screen, trace_col, False, pts, 2)

        y_cursor += 128

        # Section 3: Interactive Controls Reference
        card3 = pygame.Rect(x_pad - 6, y_cursor, SIDEBAR_WIDTH - 28, 285)
        pygame.draw.rect(screen, COLOR_CARD, card3, border_radius=5)
        pygame.draw.rect(screen, COLOR_BORDER, card3, width=1, border_radius=5)
        y_cursor += 10

        sec3 = fonts["section"].render("INTERACTIVE CONTROLS & ANALYTICS", True, COLOR_EXC)
        screen.blit(sec3, (x_pad + 4, y_cursor))
        y_cursor += 22

        controls_list = [
            ("L-CLICK HOLD", "Excite & select neuron (+I_click)"),
            ("[RIGHT ARR]", "Single-step + animate synaptic pulses"),
            ("[UP / DOWN]", "Ultra-slow (0.02x) to Fast (12x)"),
            ("[SPACE]", "Toggle Pause / Resume simulation"),
            ("[P] KEY", "Open 4-Panel Analytical Plots (1000 ms)"),
            ("[Z] / [X]", "Random AI reset [Z] / Quiescent [X]"),
            ("[Q / A]", "Increase / Decrease connectivity p"),
            ("[W/S] [E/D]", "Adjust J_EE and J_EI weights"),
            ("[R/F] [T/G]", "Adjust J_IE and J_II weights"),
            ("[Y/H] [ESC]", "Adjust I_ext [Y/H] / Return [ESC]"),
        ]
        for key_txt, desc_txt in controls_list:
            highlight = ("P" in key_txt) or ("RIGHT" in key_txt)
            k_surf = fonts["small"].render(key_txt, True, COLOR_EXC if highlight else COLOR_TEXT)
            d_surf = fonts["small"].render(desc_txt, True, COLOR_TEXT_DIM)
            screen.blit(k_surf, (x_pad + 4, y_cursor))
            screen.blit(d_surf, (x_pad + 102, y_cursor))
            y_cursor += 23

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()