import pygame
import sys
import numpy as np
import math
import os
import argparse
import copy
from env import GridWorldFireGame
from heuristic_agent import HeuristicAgent


# --- Core Configuration ---
GRID_SIZE = 15

HUD_WIDTH = 320
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 650

# ---- Isometric layout ----
ARENA_WIDTH = SCREEN_WIDTH - HUD_WIDTH
SAFE_MARGIN = 45

TILE_SIZE = int((ARENA_WIDTH - (SAFE_MARGIN * 2)) / (GRID_SIZE - 1) / 2)
ISO_DEPTH_SCALE = 0.5

ISO_OFFSET_X = ARENA_WIDTH // 2
ISO_OFFSET_Y = 150 + (GRID_SIZE * 2)

# --- Cyberpunk Neon Palette ---
COLOR_BG = (5, 6, 10)  # Deep space night sky
COLOR_PANEL = (14, 16, 24)  # Matte control panel overlay
COLOR_P1 = (255, 46, 99)  # Neon Burst Crimson (Player 1)
COLOR_P2 = (0, 225, 255)  # Cyber Future Cyan (Player 2)
COLOR_GOLD = (255, 215, 0)  # Quantum Energy Gold (Booster)
COLOR_TEXT_MAIN = (240, 243, 255)
COLOR_LABEL = (140, 145, 160)

# Path Guide Colors
COLOR_SAFE = (0, 255, 150)
COLOR_WARN = (255, 215, 0)
COLOR_WALL = (255, 46, 99)

# 3D Block Face Shading Profiles (Top, Left, Right faces to generate depth)
BLOCK_SHADES = [
    {"top": (16, 22, 32), "left": (12, 16, 24), "right": (8, 12, 18)},  # Level 0
    {"top": (25, 35, 50), "left": (20, 28, 40), "right": (15, 21, 30)},  # Level 1
    {"top": (35, 50, 72), "left": (28, 40, 58), "right": (21, 30, 44)},  # Level 2
    {"top": (45, 68, 96), "left": (36, 54, 76), "right": (27, 40, 58)},  # Level 3
    {"top": (58, 88, 122), "left": (46, 70, 98), "right": (34, 52, 74)},  # Level 4
    {"top": (72, 110, 150), "left": (58, 88, 120), "right": (44, 66, 90)}  # Level 5
]

ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY", None: "—"}


class FireGameUI:
    def __init__(self, env, player_1_agent=None, player_2_agent=None, total_matches=1, p1_name="PLAYER 01",
                 p2_name="PLAYER 02", step_mode='automatic'):
        pygame.init()
        self.env = env
        self.agents = {1: player_1_agent, 2: player_2_agent}
        self.player_names = {1: p1_name, 2: p2_name}
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("📐 Fire Trail: 3D Isometric Cyber Arena")

        font_name = "Century Gothic" if "centurygothic" in pygame.font.get_fonts() else "Arial"
        self.title_font = pygame.font.SysFont(font_name, 24, bold=True)
        self.main_font = pygame.font.SysFont(font_name, 16, bold=True)
        self.std_font = pygame.font.SysFont(font_name, 13, bold=True)
        self.digit_font = pygame.font.SysFont("Impact", 16)
        self.ko_font = pygame.font.SysFont("Impact", 42, italic=True)

        self.clock = pygame.time.Clock()
        self.reset_btn_rect = pygame.Rect(SCREEN_WIDTH - HUD_WIDTH + 40, SCREEN_HEIGHT - 65, 240, 45)
        self.pending = {1: None, 2: None}
        self.frame_count = 0
        self.step_mode = step_mode

        self.total_matches = total_matches
        self.current_match_idx = 1
        self.reported = False

        self.booster_flash_timer = 0
        self.last_booster_counts = {1: 0, 2: 0}
        self.booster_absorb_player = None

        self.tournament_results = {
            "p1_wins": 0, "p2_wins": 0, "draws": 0, "total_turns": 0,
            1: {"scores": [], "wall_hits": 0, "cliff_hits": 0, "fire_hits": 0, "collisions": 0, "own_conversions": 0,
                "enemy_conversions": 0, "boosters": 0},
            2: {"scores": [], "wall_hits": 0, "cliff_hits": 0, "fire_hits": 0, "collisions": 0, "own_conversions": 0,
                "enemy_conversions": 0, "boosters": 0}
        }

    def is_human(self, p_id):
        return self.agents[p_id] is None

    def player_type_label(self, p_id):
        agent = self.agents[p_id]
        if agent is None: return "HUMAN"
        class_name = agent.__class__.__name__
        if class_name == "NetworkAgent": return "MODEL"
        if class_name == "HeuristicAgent": return "HEURISTIC"
        return class_name.upper()

    def fill_agent_actions(self, state, info):
        """Query each agent once per step."""
        for p_id in [1, 2]:
            agent = self.agents[p_id]
            if agent is None:
                continue
            if self.pending[p_id] is None:
                self.pending[p_id] = int(agent.select_action(state, info))

    def to_iso(self, r, c, h=0):
        """Convert grid (row, col, height) to isometric screen coordinates."""
        iso_x = (c - r) * TILE_SIZE + ISO_OFFSET_X
        iso_y = (c + r) * TILE_SIZE * ISO_DEPTH_SCALE + ISO_OFFSET_Y
        iso_y -= h * 14  # Vertical extrusion for topography contrast
        return int(iso_x), int(iso_y)

    def draw_text_clear(self, text, font, color, x, y, shadow=True):
        if shadow:
            shadow_surf = font.render(text, True, (2, 4, 8))
            self.screen.blit(shadow_surf, (int(x) + 1, int(y) + 1))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, (int(x), int(y)))

    def alpha(self, value):
        return max(0, min(255, int(value)))

    def draw_preparation_arrow(self, center, action, color):
        """Draws direction locks on top of the entity vectors when an action is registered."""
        cx, cy = center
        pts = []
        size = 14
        offset = 24

        if action == 0:
            pts = [(cx, cy - offset - size), (cx - 10, cy - offset), (cx + 10, cy - offset)]
        elif action == 1:
            pts = [(cx, cy + offset + size), (cx - 10, cy + offset), (cx + 10, cy + offset)]
        elif action == 2:
            pts = [(cx - offset - size, cy), (cx - offset, cy - 10), (cx - offset, cy + 10)]
        elif action == 3:
            pts = [(cx + offset + size, cy), (cx + offset, cy - 10), (cx + offset, cy + 10)]

        if pts:
            pygame.draw.polygon(self.screen, (255, 255, 255), pts)
            pygame.draw.polygon(self.screen, color, pts, 2)

    def draw_iso_block(self, r, c, h_val):
        """Render a single 3D block at grid position (r, c) with elevation h_val."""
        shades = BLOCK_SHADES[h_val]
        t1 = self.to_iso(r, c, h_val)
        t2 = self.to_iso(r, c + 1, h_val)
        t3 = self.to_iso(r + 1, c + 1, h_val)
        t4 = self.to_iso(r + 1, c, h_val)
        b2 = self.to_iso(r, c + 1, 0)
        b3 = self.to_iso(r + 1, c + 1, 0)
        b4 = self.to_iso(r + 1, c, 0)

        pygame.draw.polygon(self.screen, shades["left"], [t4, t3, (b3[0], b3[1]), (b4[0], b4[1])])
        pygame.draw.polygon(self.screen, shades["right"], [t2, t3, (b3[0], b3[1]), (b2[0], b2[1])])
        pygame.draw.polygon(self.screen, shades["top"], [t1, t2, t3, t4])

        # Lower tiles get higher-contrast edges to avoid blending into the background.
        wire_alpha = max(10, 45 - (h_val * 6))
        wire_color = (shades["top"][0] + wire_alpha, shades["top"][1] + wire_alpha,
                      shades["top"][2] + int(wire_alpha * 1.2))
        pygame.draw.polygon(self.screen, wire_color, [t1, t2, t3, t4], 1)


    def draw_player_card(self, x, y, p_id, info, act):
        color = COLOR_P1 if p_id == 1 else COLOR_P2
        card_rect = pygame.Rect(x, y, HUD_WIDTH - 40, 115)

        if self.booster_flash_timer > 0 and self.booster_absorb_player == p_id:
            panel_fill = (22, 26, 38)
            panel_border = COLOR_GOLD
        else:
            panel_fill = (18, 20, 30)
            panel_border = (32, 38, 52)

        pygame.draw.rect(self.screen, panel_fill, card_rect, border_radius=12)
        pygame.draw.rect(self.screen, panel_border, card_rect, 1, border_radius=12)
        pygame.draw.rect(self.screen, color, (x, y + 15, 4, 85), border_radius=2)

        self.draw_text_clear(self.player_names[p_id], self.main_font, color, x + 15, y + 12)

        status_str = self.player_type_label(p_id)
        status_clr = (0, 255, 150)
        self.draw_text_clear(status_str, self.std_font, status_clr, x + 140, y + 15)

        pygame.draw.rect(self.screen, (8, 10, 14), (x + 15, y + 42, 220, 8), border_radius=4)
        lives_left = info.get('lives_left', 0)
        lives_ratio = max(0, lives_left / self.env.starting_lives)
        if lives_ratio > 0:
            pygame.draw.rect(self.screen, color, (x + 15, y + 42, int(220 * lives_ratio), 8), border_radius=4)

        # Stats display
        self.draw_text_clear(f"LIVES: {lives_left}", self.std_font, color, x + 15, y + 62)
        self.draw_text_clear(f"SCORE: {int(info.get('total_reward', 0))}", self.std_font, COLOR_TEXT_MAIN, x + 15,
                             y + 85)
        self.draw_text_clear(f"LOC: {info.get('position', (0, 0))}", self.std_font, (140, 150, 180), x + 125, y + 85)

    def draw_grid_3d(self, state, info):
        """Render the full 3D isometric grid with fire trails, boosters, and player tokens."""
        terrain, players, fire, boosters = state[0], state[1], state[2], state[3]

        human_pos = None
        human_id = None
        for p_id in [1, 2]:
            if self.is_human(p_id):
                human_pos = self.env.player_positions.get(p_id)
                human_id = p_id

            current_booster_count = info['players'][p_id].get('num_powerups_eaten', 0)
            if current_booster_count > self.last_booster_counts[p_id] and self.frame_count > 5:
                self.booster_flash_timer = 15  # Runs subtle card outline highlight frames
                self.booster_absorb_player = p_id
            self.last_booster_counts[p_id] = current_booster_count

        if self.booster_flash_timer > 0:
            self.booster_flash_timer -= 1

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                h_val = int(terrain[r, c])

                self.draw_iso_block(r, c, h_val)

                t1 = self.to_iso(r, c, h_val)
                t2 = self.to_iso(r, c + 1, h_val)
                t3 = self.to_iso(r + 1, c + 1, h_val)
                t4 = self.to_iso(r + 1, c, h_val)
                center_x = (t1[0] + t3[0]) // 2
                center_y = (t1[1] + t3[1]) // 2

                # Highlight adjacent tiles to guide the human player.
                if human_pos is not None:
                    hr, hc = human_pos
                    if abs(r - hr) + abs(c - hc) == 1:
                        curr_h = terrain[hr, hc]
                        diff = h_val - curr_h

                        is_enemy_fire = (fire[r, c] < 0) if human_id == 1 else (fire[r, c] > 0)

                        if diff > 1:
                            risk_color = COLOR_WALL
                        elif diff < -1 or is_enemy_fire:
                            risk_color = COLOR_WARN
                        else:
                            risk_color = COLOR_SAFE

                        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                        pygame.draw.polygon(s, (*risk_color, 45), [t1, t2, t3, t4])  # Translucent Fill
                        pygame.draw.polygon(s, (*risk_color, 200), [t1, t2, t3, t4], 2)  # High Contrast Border
                        self.screen.blit(s, (0, 0))

                # Fire trail
                if fire[r, c] != 0:
                    f_val = fire[r, c]
                    f_color = COLOR_P1 if f_val > 0 else COLOR_P2
                    timer_ratio = abs(f_val) / 10.0

                    s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

                    if int(abs(f_val)) == 1:
                        # Pulse on the final timer tick.
                        pulse_intensity = int(110 + math.sin(self.frame_count * 0.4) * 50)
                        trail_alpha = pulse_intensity
                        base_fill_color = (255, 0, 0)
                        digit_color = (255, 50, 50)
                    else:
                        trail_alpha = int(35 + (timer_ratio * 70))
                        base_fill_color = f_color
                        digit_color = (240, 245, 255)

                    pygame.draw.polygon(s, (*base_fill_color, self.alpha(trail_alpha)), [t1, t2, t3, t4])
                    pygame.draw.polygon(s, (255, 255, 255, self.alpha(40 + timer_ratio * 100)), [t1, t2, t3, t4], 1)
                    self.screen.blit(s, (0, 0))

                    digit_surf = self.digit_font.render(str(int(abs(f_val))), True, digit_color)
                    self.screen.blit(digit_surf, digit_surf.get_rect(center=(center_x, center_y)))

                # Booster sphere
                if boosters[r, c] > 0.5:
                    shadow_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                    shadow_scale = 6 + int(math.sin(self.frame_count * 0.2) * 1.5)
                    pygame.draw.ellipse(shadow_surf, (0, 0, 0, 130),
                                        (center_x - shadow_scale, center_y - (shadow_scale // 2), shadow_scale * 2,
                                         shadow_scale))
                    self.screen.blit(shadow_surf, (0, 0))

                    float_y = center_y - 12 + int(math.sin(self.frame_count * 0.2) * 3)
                    pygame.draw.circle(self.screen, COLOR_GOLD, (center_x, float_y), 7)
                    pygame.draw.circle(self.screen, (255, 255, 255), (center_x - 2, float_y - 2), 2)

                # Player token
                player_val = players[r, c]
                if player_val != 0:
                    p_id = 1 if player_val > 0 else 2
                    color = COLOR_P1 if p_id == 1 else COLOR_P2
                    agent_y = center_y - 14

                    s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                    pygame.draw.ellipse(s, (*color, 80), (center_x - 16, center_y - 8, 32, 16), 2)
                    self.screen.blit(s, (0, 0))

                    pygame.draw.circle(self.screen, (10, 12, 18), (center_x, agent_y), 12)
                    pygame.draw.circle(self.screen, color, (center_x, agent_y), 12, 2)
                    pygame.draw.circle(self.screen, (255, 255, 255), (center_x, agent_y), 4)

                    lives_left = max(1, round(abs(player_val) * self.env.starting_lives))
                    lives_surf = self.std_font.render(f"H{lives_left}", True, (255, 255, 255))
                    self.screen.blit(lives_surf, (center_x - 8, agent_y - 24))

                    if self.pending[p_id] is not None:
                        self.draw_preparation_arrow((center_x, center_y), self.pending[p_id], color)

                        # Dynamic vector tail connecting current to intended target block
                        dr, dc = self.env.ACTIONS[self.pending[p_id]]
                        target_r, target_c = max(0, min(GRID_SIZE - 1, r + dr)), max(0, min(GRID_SIZE - 1, c + dc))
                        target_iso_x, target_iso_y = self.to_iso(target_r, target_c, int(terrain[target_r, target_c]))

                        trail_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                        pygame.draw.line(trail_surf, (*color, 140), (center_x, agent_y),
                                         (target_iso_x, target_iso_y - 14), 2)
                        self.screen.blit(trail_surf, (0, 0))


    def draw_hud(self, info):
        hud_x = SCREEN_WIDTH - HUD_WIDTH
        pygame.draw.rect(self.screen, COLOR_PANEL, (hud_x, 0, HUD_WIDTH, SCREEN_HEIGHT))

        self.draw_text_clear("MATCH OVERVIEW", self.title_font, (0, 255, 150), hud_x + 25, 25)
        self.draw_text_clear(f"TURNS: {info.get('turn_count', 0)}", self.std_font, (150, 160, 180), hud_x + 25, 60)

        self.draw_player_card(hud_x + 20, 90, 1, info['players'][1], self.pending[1])
        self.draw_player_card(hud_x + 20, 215, 2, info['players'][2], self.pending[2])

        y_booster = 345
        pygame.draw.circle(self.screen, COLOR_GOLD, (hud_x + 35, y_booster + 10), 8)
        self.draw_text_clear("BOOSTER: Resets Fire Timers", self.std_font, COLOR_GOLD, hud_x + 55, y_booster)

        y_leg = 385
        self.draw_text_clear("MOVEMENT PREVIEW", self.main_font, (0, 255, 200), hud_x + 25, y_leg)
        diag_items = [
            (COLOR_SAFE, "Stable Path: Can Pass Safely"),
            (COLOR_WARN, "Cliff/Fire: Lose One Live"),
            (COLOR_WALL, "Blocked: Can't Pass")
        ]
        for i, (clr, txt) in enumerate(diag_items):
            pygame.draw.rect(self.screen, clr, (hud_x + 25, y_leg + 45 + i * 28, 14, 14), 2)
            self.draw_text_clear(txt, self.std_font, COLOR_TEXT_MAIN, hud_x + 50, y_leg + 43 + i * 28)

        m_pos = pygame.mouse.get_pos()
        btn_hover = self.reset_btn_rect.collidepoint(m_pos)
        btn_clr = (0, 255, 150) if btn_hover else (0, 150, 100)

        pygame.draw.rect(self.screen, (20, 25, 35), self.reset_btn_rect, border_radius=12)
        pygame.draw.rect(self.screen, btn_clr, self.reset_btn_rect, 2, border_radius=12)
        btn_txt = self.main_font.render("REBOOT SESSION", True, btn_clr)
        self.screen.blit(btn_txt, btn_txt.get_rect(center=self.reset_btn_rect.center))

    def _accumulate_match_metrics(self, info):
        winner = info.get("winner")
        if winner == 1:
            self.tournament_results["p1_wins"] += 1
        elif winner == 2:
            self.tournament_results["p2_wins"] += 1
        else:
            self.tournament_results["draws"] += 1

        self.tournament_results["total_turns"] += info.get("turn_count", 0)

        for p_id in [1, 2]:
            p_data = info["players"][p_id]
            r_ledger = self.tournament_results[p_id]
            r_ledger["scores"].append(p_data.get("total_reward", 0))
            r_ledger["boosters"] += p_data.get("num_powerups_eaten", 0)
            r_ledger["wall_hits"] += p_data.get("num_wall_or_border_hits", 0)
            r_ledger["cliff_hits"] += p_data.get("num_cliff_hits", 0)
            r_ledger["fire_hits"] += p_data.get("num_enemy_fire_hits", 0)
            r_ledger["collisions"] += p_data.get("num_player_collision_hits", 0)
            r_ledger["own_conversions"] += p_data.get("num_own_fire_conversions", 0)
            r_ledger["enemy_conversions"] += p_data.get("num_enemy_fire_conversions", 0)

    def print_tournament_summary(self):
        total = self.total_matches
        res = self.tournament_results
        print("\n" + "=" * 70)
        print(f" 🏆  TOURNAMENT FINAL BENCHMARK METRICS REPORT ({total} MATCHES)")
        print("=" * 70)
        print("  Outcome Breakdown:")
        print(f"  🔹 Player 1 Wins : {res['p1_wins']:>4} ({res['p1_wins'] / total * 100:.1f}%)")
        print(f"  🔹 Player 2 Wins : {res['p2_wins']:>4} ({res['p2_wins'] / total * 100:.1f}%)")
        print(f"  🔹 Total Draws   : {res['draws']:>4} ({res['draws'] / total * 100:.1f}%)")
        print(f"  🔹 Average Turns : {res['total_turns'] / total:>4.1f} turns/match")
        print("-" * 70)
        print("  🧠 AGGREGATED BEHAVIORAL METRICS   |  PLAYER 01        |  PLAYER 02")
        print("-" * 70)
        print(
            f"  Avg Match Score                    |  {np.mean(res[1]['scores']):>18.1f} |  {np.mean(res[2]['scores']):>15.1f}")
        print(
            f"  Max Match Score                    |  {np.max(res[1]['scores']):>18} |  {np.max(res[2]['scores']):>15}")
        print(f"  Total Boosters Collected           |  {res[1]['boosters']:>18} |  {res[2]['boosters']:>15}")
        print(f"  Total Wall/Border Collisions       |  {res[1]['wall_hits']:>18} |  {res[2]['wall_hits']:>15}")
        print(f"  Total Cliff Drop Hazards           |  {res[1]['cliff_hits']:>18} |  {res[2]['cliff_hits']:>15}")
        print(f"  Total Enemy Fire Stepped On        |  {res[1]['fire_hits']:>18} |  {res[2]['fire_hits']:>15}")
        print(f"  Total Head-on Collisions           |  {res[1]['collisions']:>18} |  {res[2]['collisions']:>15}")
        print(
            f"  Total Own Fire Extinguished        |  {res[1]['own_conversions']:>18} |  {res[2]['own_conversions']:>15}")
        print(
            f"  Total Enemy Fire Converted         |  {res[1]['enemy_conversions']:>18} |  {res[2]['enemy_conversions']:>15}")
        print("=" * 70 + "\n")

    def run(self):
        is_headless = pygame.display.get_driver() == "dummy"
        state, info, done = self.env.reset(), self.env.info(), False
        self.reported = False
        history = []  # list of (env_deepcopy, done_flag) — used in manual step mode

        while True:
            self.frame_count += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN and self.reset_btn_rect.collidepoint(event.pos):
                    state = self.env.reset()
                    info = self.env.info()
                    done = False
                    self.pending = {1: None, 2: None}
                    self.reported = False
                    history.clear()

                if event.type == pygame.KEYDOWN:
                    if self.step_mode == 'manual':
                        if event.key == pygame.K_RIGHT and not done:
                            if self.pending[1] is not None and self.pending[2] is not None:
                                history.append((copy.deepcopy(self.env), done))
                                state, _, _, done, info = self.env.step(self.pending[1], self.pending[2])
                                self.pending = {1: None, 2: None}
                        elif event.key == pygame.K_LEFT and history:
                            env_snap, prev_done = history.pop()
                            self.env = env_snap
                            state = self.env.get_state()
                            info = self.env.info()
                            done = prev_done
                            self.pending = {1: None, 2: None}

                    if not done:
                        keys = {
                            pygame.K_w: (1, 0), pygame.K_s: (1, 1),
                            pygame.K_a: (1, 2), pygame.K_d: (1, 3),
                            pygame.K_UP: (2, 0), pygame.K_DOWN: (2, 1),
                            pygame.K_LEFT: (2, 2), pygame.K_RIGHT: (2, 3),
                        }
                        if event.key in keys:
                            p_id, action = keys[event.key]
                            if self.is_human(p_id):
                                self.pending[p_id] = action

            if not done:
                self.fill_agent_actions(state, info)

            if self.step_mode == 'automatic':
                if not done and self.pending[1] is not None and self.pending[2] is not None:
                    state, _, _, done, info = self.env.step(self.pending[1], self.pending[2])
                    self.pending = {1: None, 2: None}

            self.screen.fill(COLOR_BG)
            self.draw_grid_3d(state, info)
            self.draw_hud(info)

            if done:
                s = pygame.Surface((SCREEN_WIDTH - HUD_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                s.fill((5, 7, 14, 160))
                pygame.draw.polygon(s, (14, 18, 28, 235), [(0, 0), (SCREEN_WIDTH - HUD_WIDTH, 120),
                                                           (SCREEN_WIDTH - HUD_WIDTH, SCREEN_HEIGHT),
                                                           (0, SCREEN_HEIGHT - 120)])
                self.screen.blit(s, (0, 0))

                w_id = info.get('winner')
                winner_msg = f"PLAYER 0{w_id} SECURED VICTORY" if w_id else "CRITICAL DEADLOCK: DRAW"
                text_color = COLOR_P1 if w_id == 1 else (COLOR_P2 if w_id == 2 else COLOR_LABEL)

                self.draw_text_clear("TERMINAL STATUS REPORT", self.main_font, COLOR_LABEL, 40, SCREEN_HEIGHT // 2 - 60)
                self.draw_text_clear(winner_msg, self.ko_font, text_color, 40, SCREEN_HEIGHT // 2 - 30)

                if not self.reported:
                    self._accumulate_match_metrics(info)
                    self.reported = True

                    if self.current_match_idx < self.total_matches:
                        self.current_match_idx += 1
                        state = self.env.reset()
                        info = self.env.info()
                        done = False
                        self.pending = {1: None, 2: None}
                        self.reported = False
                    else:
                        self.print_tournament_summary()

            pygame.display.flip()
            self.clock.tick(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fire Trail: 3D Cyber Arena Interface")
    parser.add_argument("--matches", "-m", type=int, default=1, help="Total matches to execute (default: 1)")
    parser.add_argument("--headless", action="store_true", help="Execute in headless background driver mode")
    args = parser.parse_args()

    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    game_env = GridWorldFireGame(grid_size=GRID_SIZE, player_lives=3, booster_prob=0.001, debug=False)

    agent2 = HeuristicAgent(player_id=2)
    ui = FireGameUI(
        game_env,
        player_1_agent=None,
        player_2_agent=agent2,
        total_matches=args.matches,
        p1_name="PLAYER 01",
        p2_name="PLAYER 02"
    )
    ui.total_matches = args.matches
    ui.run()
