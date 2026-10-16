#!/usr/bin/env python3
"""
main.py - Neural Network Sandbox Launcher
A minimalist Dark Grey & Magenta central dashboard for launching interactive
computational neuroscience simulations.
"""

import os
import sys
import math
import ctypes
import subprocess
import pygame

# Lock Windows DPI awareness BEFORE Pygame or Matplotlib initializes so window
# dimensions never shrink or rescale across monitor DPI changes.


# =============================================================================
# WINDOW & THEME CONFIGURATION (Adjustable Resolution)
# =============================================================================
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60

# Minimalist Dark Grey & Magenta Palette
COLOR_BG = (18, 18, 18)             # #121212 Very Dark Grey
COLOR_PANEL = (30, 30, 30)          # #1E1E1E Dark Grey UI Panel
COLOR_PANEL_HOVER = (45, 45, 45)    # #2D2D2D Lighter Panel on Hover
COLOR_BORDER = (60, 60, 60)         # Subtle Border Grey
COLOR_MAGENTA = (255, 0, 127)       # #FF007F Primary Accent (Excitatory)
COLOR_MAGENTA_DIM = (140, 0, 70)    # Dimmed Magenta for inactive accents
COLOR_SECONDARY = (216, 226, 236)   # Crisp Ice-Silver Contrast (Inhibitory)
COLOR_TEXT = (224, 224, 224)        # #E0E0E0 Crisp Light Grey
COLOR_TEXT_MUTED = (130, 130, 130)  # Muted Grey for descriptions


class ModuleButton:
    """Interactive dashboard card representing a simulation module."""

    def __init__(self, rect, title, subtitle, tags, script_name=None, enabled=True):
        self.rect = pygame.Rect(rect)
        self.title = title
        self.subtitle = subtitle
        self.tags = tags
        self.script_name = script_name
        self.enabled = enabled
        self.hover_progress = 0.0

    def update(self, mouse_pos, dt):
        is_hovered = self.rect.collidepoint(mouse_pos) and self.enabled
        target = 1.0 if is_hovered else 0.0
        self.hover_progress += (target - self.hover_progress) * min(1.0, dt * 12.0)

    def draw(self, surface, fonts):
        bg_color = tuple(
            int(COLOR_PANEL[i] + (COLOR_PANEL_HOVER[i] - COLOR_PANEL[i]) * self.hover_progress)
            for i in range(3)
        )
        if self.enabled:
            border_color = tuple(
                int(COLOR_BORDER[i] + (COLOR_MAGENTA[i] - COLOR_BORDER[i]) * self.hover_progress)
                for i in range(3)
            )
        else:
            border_color = (38, 38, 38)

        pygame.draw.rect(surface, bg_color, self.rect, border_radius=6)
        pygame.draw.rect(surface, border_color, self.rect, width=2, border_radius=6)

        accent_rect = pygame.Rect(self.rect.x, self.rect.y, 6, self.rect.height)
        accent_color = COLOR_MAGENTA if self.enabled else (55, 55, 55)
        pygame.draw.rect(
            surface,
            accent_color,
            accent_rect,
            border_top_left_radius=6,
            border_bottom_left_radius=6,
        )

        title_color = COLOR_TEXT if self.enabled else COLOR_TEXT_MUTED
        title_surf = fonts["card_title"].render(self.title, True, title_color)
        surface.blit(title_surf, (self.rect.x + 24, self.rect.y + 20))

        badge_text = "READY TO LAUNCH" if self.enabled else "LOCKED / PLACEHOLDER"
        badge_bg = COLOR_MAGENTA if self.enabled else (42, 42, 42)
        badge_fg = COLOR_BG if self.enabled else COLOR_TEXT_MUTED
        badge_surf = fonts["badge"].render(f" {badge_text} ", True, badge_fg)
        badge_rect = badge_surf.get_rect(
            topright=(self.rect.right - 20, self.rect.y + 22)
        )
        pygame.draw.rect(surface, badge_bg, badge_rect.inflate(10, 6), border_radius=3)
        surface.blit(badge_surf, badge_rect)

        sub_surf = fonts["body"].render(self.subtitle, True, COLOR_TEXT_MUTED)
        surface.blit(sub_surf, (self.rect.x + 24, self.rect.y + 56))

        tag_x = self.rect.x + 24
        tag_y = self.rect.bottom - 34
        for tag in self.tags:
            tag_surf = fonts["badge"].render(
                tag, True, COLOR_TEXT if self.enabled else COLOR_TEXT_MUTED
            )
            tag_box = tag_surf.get_rect(topleft=(tag_x, tag_y)).inflate(12, 6)
            pygame.draw.rect(surface, (22, 22, 22), tag_box, border_radius=3)
            pygame.draw.rect(
                surface,
                COLOR_MAGENTA_DIM if self.enabled else (40, 40, 40),
                tag_box,
                width=1,
                border_radius=3,
            )
            surface.blit(tag_surf, (tag_x + 6, tag_y + 3))
            tag_x += tag_box.width + 10


def draw_decorative_network(surface, t):
    """Draws a subtle animated spiking graph motif in the right-hand preview pane."""
    center_x, center_y = 950, 430
    radius = 155
    num_nodes = 18
    nodes = []

    for i in range(num_nodes):
        angle = (2 * math.pi * i / num_nodes) + t * 0.15
        r_mod = radius * (0.65 if i % 4 == 0 else 1.0)
        nx = center_x + math.cos(angle) * r_mod
        ny = center_y + math.sin(angle) * r_mod
        is_inh = (i % 4 == 0)
        nodes.append((nx, ny, is_inh))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (i * 7 + j * 3) % 5 == 0:
                x1, y1, _ = nodes[i]
                x2, y2, _ = nodes[j]
                pygame.draw.aaline(surface, (42, 42, 42), (x1, y1), (x2, y2))

    for idx, (nx, ny, is_inh) in enumerate(nodes):
        pulse = 0.5 + 0.5 * math.sin(t * 3.5 + idx * 1.1)
        if is_inh:
            pygame.draw.circle(
                surface, COLOR_SECONDARY, (int(nx), int(ny)), int(5 + 2 * pulse), width=2
            )
        else:
            c = (int(90 + 165 * pulse), 0, int(45 + 82 * pulse))
            pygame.draw.circle(surface, c, (int(nx), int(ny)), int(4 + 3 * pulse))


def launch_subgame(script_filename, status_callback):
    """
    Cleanly closes the launcher display while the sub-game runs, then recreates
    and focuses the launcher window immediately upon sub-game exit.
    """
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_filename)
    if not os.path.exists(script_path):
        status_callback(f"ERROR: Could not find '{script_filename}' in directory.")
        return pygame.display.get_surface()

    # Close the launcher window rather than minimizing it so it never stays stuck minimized
    pygame.display.quit()
    try:
        subprocess.run([sys.executable, script_path], check=False)
    except Exception as exc:
        status_callback(f"LAUNCH ERROR: {exc}")

    # Re-initialize the display in the foreground
    pygame.display.init()
    pygame.display.set_caption("NEURAL NETWORK SANDBOX // CENTRAL LAUNCHER")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.event.clear()

    # Force window to the foreground on Windows
    if sys.platform == "win32":
        try:
            hwnd = pygame.display.get_wm_info().get("window")
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception:
            pass

    status_callback("READY: Returned from simulation module.")
    return screen


def main():
    pygame.init()
    pygame.display.set_caption("NEURAL NETWORK SANDBOX // CENTRAL LAUNCHER")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    font_names = ["consolas", "menlo", "dejavusansmono", "couriernew", "monospace"]
    fonts = {
        "header": pygame.font.SysFont(font_names, 28, bold=True),
        "subheader": pygame.font.SysFont(font_names, 14, bold=False),
        "card_title": pygame.font.SysFont(font_names, 19, bold=True),
        "body": pygame.font.SysFont(font_names, 14, bold=False),
        "badge": pygame.font.SysFont(font_names, 12, bold=True),
    }

    buttons = [
        ModuleButton(
            rect=(60, 170, 640, 125),
            title="Game 1: E-I LIF Network",
            subtitle="Erdos-Renyi Leaky Integrate-and-Fire balanced cortical network.",
            tags=["N=100 E/I POPULATIONS", "ULTRA-SLOW & STEP", "4-PANEL ANALYTICS"],
            script_name="game1_lif_network.py",
            enabled=True,
        ),
        ModuleButton(
            rect=(60, 315, 640, 125),
            title="Game 2: Hodgkin-Huxley Conductance Model",
            subtitle="Voltage-gated Na+/K+ ion channel kinetics and phase-plane bifurcations.",
            tags=["4D ODE SYSTEM", "VOLTAGE CLAMP", "PHASE PORTRAIT"],
            script_name=None,
            enabled=False,
        ),
        ModuleButton(
            rect=(60, 460, 640, 125),
            title="Game 3: Hopfield Attractor & Associative Memory",
            subtitle="Energy landscape minimization, Hebbian storage, and pattern completion.",
            tags=["LYAPUNOV ENERGY", "HEBBIAN WEIGHTS", "THERMAL ANNEALING"],
            script_name=None,
            enabled=False,
        ),
        ModuleButton(
            rect=(60, 605, 640, 125),
            title="Game 4: STDP Synaptic Plasticity Sandbox",
            subtitle="Spike-Timing-Dependent Plasticity and emergent synfire chain assembly.",
            tags=["ASYMMETRIC HEBBIAN", "WEIGHT DRIFT", "SYNFIRE CHAINS"],
            script_name=None,
            enabled=False,
        ),
    ]

    status_message = "SYSTEM READY // SELECT AN ACTIVE MODULE TO INITIALIZE"
    time_elapsed = 0.0

    def set_status(msg):
        nonlocal status_message
        status_message = msg

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        time_elapsed += dt
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    screen = launch_subgame("game1_lif_network.py", set_status)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for btn in buttons:
                    if btn.rect.collidepoint(event.pos):
                        if btn.enabled and btn.script_name:
                            screen = launch_subgame(btn.script_name, set_status)
                        else:
                            set_status(
                                f"NOTICE: '{btn.title}' is a placeholder for a future module."
                            )

        for btn in buttons:
            btn.update(mouse_pos, dt)

        # ---------------------------------------------------------------------
        # RENDERING
        # ---------------------------------------------------------------------
        screen.fill(COLOR_BG)

        pygame.draw.rect(screen, COLOR_PANEL, (0, 0, WINDOW_WIDTH, 115))
        pygame.draw.line(screen, COLOR_MAGENTA, (0, 115), (WINDOW_WIDTH, 115), 2)

        header_surf = fonts["header"].render(
            "NEURAL NETWORK SANDBOX // COMPUTATIONAL LAB", True, COLOR_TEXT
        )
        screen.blit(header_surf, (60, 32))

        sub_surf = fonts["subheader"].render(
            "Interactive Spiking Dynamics, Balanced E-I Networks & Electrophysiology",
            True,
            COLOR_MAGENTA,
        )
        screen.blit(sub_surf, (60, 72))

        for btn in buttons:
            btn.draw(screen, fonts)

        preview_rect = pygame.Rect(740, 170, 400, 560)
        pygame.draw.rect(screen, COLOR_PANEL, preview_rect, border_radius=6)
        pygame.draw.rect(screen, COLOR_BORDER, preview_rect, width=1, border_radius=6)

        p_title = fonts["card_title"].render("TOPOLOGY PREVIEW", True, COLOR_TEXT)
        screen.blit(p_title, (765, 195))
        p_sub = fonts["body"].render("Asynchronous Irregular (AI) Regime", True, COLOR_MAGENTA)
        screen.blit(p_sub, (765, 222))

        draw_decorative_network(screen, time_elapsed)

        legend_y = 615
        pygame.draw.circle(screen, COLOR_MAGENTA, (775, legend_y + 8), 6)
        leg_e = fonts["body"].render("Excitatory Population (80% Pyramidal)", True, COLOR_TEXT)
        screen.blit(leg_e, (792, legend_y))

        pygame.draw.circle(screen, COLOR_SECONDARY, (775, legend_y + 34), 6, width=2)
        leg_i = fonts["body"].render("Inhibitory Population (20% Interneurons)", True, COLOR_TEXT)
        screen.blit(leg_i, (792, legend_y + 26))

        hint = fonts["badge"].render(
            "SHORTCUT: PRESS [1] TO LAUNCH GAME 1", True, COLOR_TEXT_MUTED
        )
        screen.blit(hint, (765, 685))

        pygame.draw.rect(screen, (24, 24, 24), (0, WINDOW_HEIGHT - 36, WINDOW_WIDTH, 36))
        pygame.draw.line(
            screen, COLOR_BORDER, (0, WINDOW_HEIGHT - 36), (WINDOW_WIDTH, WINDOW_HEIGHT - 36), 1
        )
        status_surf = fonts["badge"].render(status_message, True, COLOR_TEXT)
        screen.blit(status_surf, (60, WINDOW_HEIGHT - 24))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()