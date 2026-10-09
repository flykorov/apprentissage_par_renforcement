"""
Étape 1 du projet RL : circuit procédural OUVERT (point A -> point B) avec
virages, conduite manuelle de test, crash définitif en sortie de piste, et
détection de la ligne d'arrivée.

Commandes :
  Flèches / ZQSD-WASD : conduire
  R                    : nouveau circuit (nouveau seed)
  ESPACE               : rejouer le même circuit (après un crash/victoire)
  ECHAP                : quitter
"""

import sys
import pygame
import numpy as np
import random


from track_generator import Track
from car import Car

WIDTH, HEIGHT = 900, 700
FPS = 60

COLOR_GRASS = (34, 120, 60)
COLOR_ROAD = (60, 60, 65)
COLOR_KERB_A = (220, 40, 40)
COLOR_KERB_B = (240, 240, 240)
COLOR_TEXT = (255, 255, 255)
COLOR_START = (60, 200, 90)
COLOR_FINISH_FLAG_A = (20, 20, 20)
COLOR_FINISH_FLAG_B = (240, 240, 240)
COLOR_SENSOR = (255, 220, 60)


def draw_sensors(surface, car, track):
    """Dessine les rayons des capteurs (debug visuel) : vert si loin du mur,
    orange/rouge si le mur est proche — un aperçu de ce que 'voit' l'agent."""
    if not car.alive:
        return
    readings = car.get_sensor_readings(track)
    points = car.get_sensor_points(track)
    for distance, point in zip(readings, points):
        ratio = min(1.0, distance / car.SENSOR_MAX_DISTANCE)
        color = (int(255 * (1 - ratio)), int(255 * ratio), 60)
        pygame.draw.line(surface, color, (car.x, car.y), point, 2)
        pygame.draw.circle(surface, color, (int(point[0]), int(point[1])), 4)


def draw_checker_line(surface, p_left, p_right, perp_dir, n_squares, thickness=6):
    for i in range(n_squares):
        t0, t1 = i / n_squares, (i + 1) / n_squares
        p0 = p_left + (p_right - p_left) * t0
        p1 = p_left + (p_right - p_left) * t1
        color = COLOR_FINISH_FLAG_B if i % 2 == 0 else COLOR_FINISH_FLAG_A
        perp = perp_dir * thickness
        poly = [p0 - perp, p1 - perp, p1 + perp, p0 + perp]
        pygame.draw.polygon(surface, color, poly)


def draw_track(surface, track: Track):
    surface.fill(COLOR_GRASS)

    pygame.draw.polygon(surface, COLOR_ROAD, track.ribbon_polygon.tolist())

    # bordures (kerbs) en pointillés rouge/blanc, sur les deux bords
    for i in range(0, len(track.left_boundary), 4):
        color = COLOR_KERB_A if (i // 4) % 2 == 0 else COLOR_KERB_B
        pygame.draw.circle(surface, color, track.left_boundary[i].astype(int), 3)
        pygame.draw.circle(surface, color, track.right_boundary[i].astype(int), 3)

    # checkpoints (discrets)
    for left_pt, right_pt in track.checkpoints:
        pygame.draw.line(surface, (255, 255, 255), left_pt, right_pt, 1)

    # ligne de départ (verte, pleine) et ligne d'arrivée (damier)
    sl, sr = track.start_line
    pygame.draw.line(surface, COLOR_START, sl, sr, 6)

    fl, fr = track.finish_line

    tangent = np.array([fr[1] - fl[1], fl[0] - fr[0]])
    norm = np.linalg.norm(tangent)
    perp_dir = tangent / norm if norm > 1e-6 else np.array([0.0, 0.0])
    draw_checker_line(surface, fl, fr, perp_dir, n_squares=6)


def reset_car(track):
    return Car(*track.start_pos, track.start_angle)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Circuit procédural A -> B - Projet RL")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    font_big = pygame.font.SysFont("consolas", 42, bold=True)

    track = Track(WIDTH, HEIGHT, track_width=70, seed=random.randint(0, 999_999))
    car = reset_car(track)
    elapsed_time = 0.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    track = Track(
                        WIDTH, HEIGHT, track_width=70, seed=random.randint(0, 999_999)
                    )
                    car = reset_car(track)
                    elapsed_time = 0.0
                elif event.key == pygame.K_SPACE:
                    car = reset_car(track)
                    elapsed_time = 0.0

        keys = pygame.key.get_pressed()
        action = Car.action_from_keys(keys)
        car.update(dt, action)

        if car.alive:
            elapsed_time += dt
            if not track.point_on_track((car.x, car.y)):
                car.crash()
            elif track.distance_to_finish((car.x, car.y)) < track.track_width / 2:
                car.finish()

        draw_track(screen, track)
        draw_sensors(screen, car, track)
        car.draw(screen)

        hud_lines = [
            f"Seed: {track.seed}   (R = nouveau circuit, ESPACE = rejouer)",
            f"Vitesse: {car.speed:5.0f} px/s     Temps: {elapsed_time:5.1f}s",
            f"Position: {car.x:.2f}, {car.y:.2f}",
            f"Sensor Distance: {car.get_sensor_distances(track)}",
            f"Sensor Point: {car.get_sensor_points(track)}",
        ]
        for i, line in enumerate(hud_lines):
            surf = font.render(line, True, COLOR_TEXT)
            screen.blit(surf, (10, 10 + i * 22))

        if car.crashed:
            msg = font_big.render("CRASH !", True, (230, 60, 60))
            sub = font.render(
                "ESPACE pour rejouer   /   R pour un nouveau circuit", True, COLOR_TEXT
            )
            screen.blit(msg, msg.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 20)))
            screen.blit(sub, sub.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 30)))
        elif car.finished:
            msg = font_big.render("VICTOIRE !", True, (60, 200, 90))
            sub = font.render(
                f"Temps : {elapsed_time:.1f}s   "
                "ESPACE pour rejouer   /   R pour un nouveau circuit",
                True,
                COLOR_TEXT,
            )
            screen.blit(msg, msg.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 20)))
            screen.blit(sub, sub.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 30)))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
