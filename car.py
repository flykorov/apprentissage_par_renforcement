"""
Voiture avec un modèle physique simple (arcade), pilotable au clavier.

Gère aussi les deux états de fin de course :
- crashed  : la voiture est sortie de la piste -> définitivement hors
             d'usage (n'écoute plus aucune touche, s'arrête net).
- finished : la voiture a franchi la ligne d'arrivée -> victoire.
"""

import math
import pygame


class Car:
    # Angles des capteurs, relatifs à l'orientation de la voiture (en
    # degrés). 0° = droit devant. C'est la "configuration des yeux" de la
    # voiture ; c'est donc ici, sur Car, que ça a du sens de vivre — Track
    # ne sait rien de la voiture, elle sait juste répondre "qu'est-ce qui
    # est touché si on envoie un rayon depuis tel point, dans telle
    # direction" (voir Track.raycast).
    SENSOR_ANGLES_DEG = [-90, -45, -20, 0, 20, 45, 90]
    SENSOR_MAX_DISTANCE = 200.0

    def __init__(self, x, y, angle=0.0):
        self.x = x
        self.y = y
        self.angle = angle  # radians
        self.speed = 0.0

        self.max_speed = 260.0
        self.acceleration = 220.0
        self.braking = 320.0
        self.friction = 90.0
        self.turn_speed = 2.6  # rad/s à vitesse max

        self.width = 18
        self.length = 30

        # États de fin de course. Une voiture "crashed" ne répond plus du
        # tout aux commandes : elle est définitivement hors d'usage.
        self.crashed = False
        self.finished = False

    @property
    def alive(self):
        """Vrai tant que la voiture est encore fonctionnelle (ni crashée, ni
        arrivée). Pratique pour l'affichage, et pour une future interface RL
        où alive=False signifie la fin de l'épisode."""
        return not self.crashed and not self.finished

    def reset(self, x, y, angle):
        self.x, self.y, self.angle = x, y, angle
        self.speed = 0.0
        self.crashed = False
        self.finished = False

    def crash(self):
        """Sortie de piste : la voiture devient définitivement inutilisable."""
        self.crashed = True
        self.speed = 0.0

    def finish(self):
        """Ligne d'arrivée franchie : victoire."""
        self.finished = True
        self.speed = 0.0

    def update(self, dt, action):
        """action = (throttle, steer), throttle in [-1,1], steer in [-1,1]
        (throttle: 1=accélère, -1=freine/recule ; steer: -1=gauche, 1=droite).
        Ne fait rien si la voiture est crashée ou a déjà fini la course.
        """
        if self.crashed or self.finished:
            return

        throttle, steer = action

        if throttle > 0:
            self.speed += self.acceleration * throttle * dt
        elif throttle < 0:
            self.speed += self.braking * throttle * dt
        else:
            if self.speed > 0:
                self.speed = max(0.0, self.speed - self.friction * dt)
            elif self.speed < 0:
                self.speed = min(0.0, self.speed + self.friction * dt)

        self.speed = max(-self.max_speed / 2, min(self.max_speed, self.speed))

        # le braquage est proportionnel à la vitesse (plus réaliste)
        turn_factor = self.speed / self.max_speed
        self.angle += steer * self.turn_speed * turn_factor * dt

        self.x += math.cos(self.angle) * self.speed * dt
        self.y += math.sin(self.angle) * self.speed * dt

    @staticmethod
    def action_from_keys(keys):
        throttle = 0.0
        steer = 0.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle += 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            throttle -= 1.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steer -= 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steer += 1.0
        return throttle, steer

    def get_corners(self):
        """Coins de la voiture (utile pour une détection de collision plus
        précise que le simple point central, si besoin plus tard)."""
        hl, hw = self.length / 2, self.width / 2
        cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
        local = [(hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)]
        return [
            (self.x + lx * cos_a - ly * sin_a, self.y + lx * sin_a + ly * cos_a)
            for lx, ly in local
        ]

    def draw(self, surface):
        if self.crashed:
            color = (90, 90, 90)
        elif self.finished:
            color = (60, 200, 90)
        else:
            color = (230, 70, 70)

        corners = self.get_corners()
        pygame.draw.polygon(surface, color, corners)

        if self.crashed:
            # petite croix pour marquer clairement l'épave
            pygame.draw.line(
                surface,
                (20, 20, 20),
                (self.x - 8, self.y - 8),
                (self.x + 8, self.y + 8),
                3,
            )
            pygame.draw.line(
                surface,
                (20, 20, 20),
                (self.x - 8, self.y + 8),
                (self.x + 8, self.y - 8),
                3,
            )
        else:
            front = (
                self.x + math.cos(self.angle) * self.length / 2,
                self.y + math.sin(self.angle) * self.length / 2,
            )
            pygame.draw.circle(
                surface, (255, 255, 255), (int(front[0]), int(front[1])), 3
            )

    def get_sensor_distances(self, track, max_distance=None):
        """Renvoie la liste des distances (une par angle de
        SENSOR_ANGLES_DEG) jusqu'au mur le plus proche dans chaque
        direction. C'est ça qui formera une partie de l'état observé par
        l'agent RL plus tard. Normalisable dans [0,1] en divisant par
        max_distance si besoin pour l'agent."""
        max_distance = max_distance or self.SENSOR_MAX_DISTANCE
        distances = []
        for rel_deg in self.SENSOR_ANGLES_DEG:
            world_angle = self.angle + math.radians(rel_deg)
            distance, _ = track.raycast((self.x, self.y), world_angle, max_distance)
            distances.append(distance)
        return distances

    def get_sensor_points(self, track, max_distance=None):
        """Comme get_sensor_readings, mais renvoie les points d'impact
        (x, y) plutôt que les distances — utile uniquement pour dessiner
        les rayons à l'écran (debug visuel)."""
        max_distance = max_distance or self.SENSOR_MAX_DISTANCE
        points = []
        for rel_deg in self.SENSOR_ANGLES_DEG:
            world_angle = self.angle + math.radians(rel_deg)
            _, point = track.raycast((self.x, self.y), world_angle, max_distance)
            points.append(point)
        return points

    # def get_sensor_distance(self, track, max_distance=None) -> list:
    #     max_distance = max_distance or self.SENSOR_MAX_DISTANCE
    #     points = []
    #     for rel_deg in self.SENSOR_ANGLES_DEG:
    #         world_angle = self.angle + math.radians(rel_deg)
    #         _, point = track.raycast((self.x, self.y), world_angle, max_distance)
    #         points.append(
    #             math.sqrt((point[0] - self.x) ** 2 + (point[1] - self.y) ** 2)
    #         )
    #     return points
