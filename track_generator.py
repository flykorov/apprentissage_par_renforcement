"""
Génération procédurale d'un circuit OUVERT (point A -> point B) avec virages.

Principe (différent d'une boucle fermée) :
1. On définit un point de départ (à gauche) et un point d'arrivée (à droite).
2. On place des points intermédiaires dont la coordonnée projetée sur l'axe
   départ->arrivée est STRICTEMENT CROISSANTE. C'est l'astuce qui garantit
   qu'une fois lissé, le tracé ne peut pas revenir en arrière sur lui-même
   (un peu comme une fonction y=f(x) qui ne peut pas se croiser elle-même).
   Le décalage latéral aléatoire de chaque point crée les virages.
3. On lisse avec une spline de Catmull-Rom OUVERTE (pas de bouclage).
4. On "offset" la centerline de +largeur/2 et -largeur/2 -> bord gauche et
   bord droit du circuit (ici pas d'ambiguïté intérieur/extérieur puisque la
   courbe est ouverte : juste un bord gauche et un bord droit).
5. On vérifie qu'aucun des deux bords ne se croise lui-même (virage trop
   serré par rapport à la largeur) ; sinon on régénère avec un nouveau seed.
6. Le premier point = ligne de DÉPART, le dernier = ligne d'ARRIVÉE.
7. Des checkpoints réguliers sont générés le long du circuit.
"""

import math
import random
import numpy as np


# ----------------------------------------------------------------------
# Géométrie de base
# ----------------------------------------------------------------------


def _catmull_rom_open(points, samples_per_segment=14):
    """Spline de Catmull-Rom OUVERTE (pas de bouclage). On ajoute des points
    fantômes aux deux extrémités (par extrapolation) pour que la courbe
    parte et arrive correctement sur le premier et le dernier point."""
    pts = np.array(points, dtype=float)
    n = len(pts)
    if n < 2:
        return pts

    phantom_start = 2 * pts[0] - pts[1]
    phantom_end = 2 * pts[-1] - pts[-2]
    extended = np.vstack([phantom_start, pts, phantom_end])

    result = []
    n_segments = len(extended) - 3
    for i in range(n_segments):
        p0, p1, p2, p3 = extended[i], extended[i + 1], extended[i + 2], extended[i + 3]
        last_segment = i == n_segments - 1
        ts = np.linspace(0, 1, samples_per_segment, endpoint=last_segment)
        for t in ts:
            t2, t3 = t * t, t * t * t
            point = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            result.append(point)

    return np.array(result)


def _offset_curve_open(curve, distance):
    """Décale une courbe OUVERTE perpendiculairement à sa tangente locale.
    Pas d'ambiguïté de sens ici (contrairement à une boucle fermée) : la
    tangente va toujours du départ vers l'arrivée, donc "distance > 0" veut
    toujours dire "le même côté" tout du long.
    """
    n = len(curve)
    offset = np.zeros_like(curve)
    for i in range(n):
        if i == 0:
            tangent = curve[1] - curve[0]
        elif i == n - 1:
            tangent = curve[-1] - curve[-2]
        else:
            tangent = curve[i + 1] - curve[i - 1]
        length = np.linalg.norm(tangent)
        if length < 1e-9:
            normal = np.array([0.0, 0.0])
        else:
            tangent = tangent / length
            normal = np.array([-tangent[1], tangent[0]])
        offset[i] = curve[i] + normal * distance
    return offset


def _is_simple_polyline(points, stride=3):
    """Détecte une auto-intersection d'une polyligne OUVERTE (pas de segment
    de fermeture entre le dernier et le premier point, contrairement à une
    boucle). Vectorisé avec numpy."""
    pts = points[::stride]
    n = len(pts)
    if n < 4:
        return True

    a = pts[:-1]
    b = pts[1:]
    m = len(a)  # nombre de segments

    p1 = a[:, None, :]
    p2 = b[:, None, :]
    p3 = a[None, :, :]
    p4 = b[None, :, :]

    def ccw(o, u, v):
        return (v[..., 1] - o[..., 1]) * (u[..., 0] - o[..., 0]) > (
            u[..., 1] - o[..., 1]
        ) * (v[..., 0] - o[..., 0])

    c1 = ccw(p1, p3, p4)
    c2 = ccw(p2, p3, p4)
    c3 = ccw(p1, p2, p3)
    c4 = ccw(p1, p2, p4)
    intersect = (c1 != c2) & (c3 != c4)

    idx = np.arange(m)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    # segments adjacents (partagent un point) : i==j, ou consécutifs
    adjacent = (ii == jj) | (jj == ii + 1) | (ii == jj + 1)
    intersect &= ~adjacent

    return not bool(intersect.any())


# ----------------------------------------------------------------------
# Génération du circuit
# ----------------------------------------------------------------------


class Track:
    MAX_ATTEMPTS = 40

    def __init__(self, width=900, height=700, track_width=70, seed=None):
        self.screen_width = width
        self.screen_height = height
        self.track_width = track_width
        self.seed = seed if seed is not None else random.randint(0, 999_999)
        self.generate()

    def generate(self):
        """Boucle itérative : on tente de générer un circuit valide (pas
        d'auto-intersection des bords) et on recommence avec un nouveau seed
        tant que ce n'est pas le cas."""
        rnd = random.Random(self.seed)

        for _ in range(self.MAX_ATTEMPTS):
            result = self._try_generate(rnd)
            if result is not None:
                self._finalize(*result)
                return
            self.seed = rnd.randint(0, 999_999)

        # dernier recours (cas extrêmement improbable) : on accepte tel quel
        self._finalize(*self._try_generate(rnd, force=True))

    def _try_generate(self, rnd, force=False):
        margin_x = self.screen_width * 0.09
        start = np.array(
            [
                margin_x,
                rnd.uniform(self.screen_height * 0.25, self.screen_height * 0.75),
            ]
        )
        end = np.array(
            [
                self.screen_width - margin_x,
                rnd.uniform(self.screen_height * 0.25, self.screen_height * 0.75),
            ]
        )

        primary = end - start
        total_len = np.linalg.norm(primary)
        primary_dir = primary / total_len
        perp_dir = np.array([-primary_dir[1], primary_dir[0]])

        n_interior = rnd.randint(4, 7)
        max_amplitude = min(self.screen_height, self.screen_width) * 0.16

        # u strictement croissant (avec un peu de gigue, mais jamais assez
        # pour inverser l'ordre) -> garantit l'absence d'auto-intersection
        # "grossière" de la centerline.
        base_us = np.linspace(0, total_len, n_interior + 2)[1:-1]
        spacing = total_len / (n_interior + 1)
        jitter_amp = spacing * 0.3
        us = [u + rnd.uniform(-jitter_amp, jitter_amp) for u in base_us]
        us = sorted(us)

        waypoints = [start]
        for u in us:
            v = rnd.uniform(-max_amplitude, max_amplitude)
            point = start + primary_dir * u + perp_dir * v
            # on garde les points dans l'écran (avec une marge pour la largeur)
            point[1] = np.clip(
                point[1], self.track_width, self.screen_height - self.track_width
            )
            waypoints.append(point)
        waypoints.append(end)

        centerline = _catmull_rom_open(waypoints, samples_per_segment=14)

        if not force and not _is_simple_polyline(centerline):
            return None

        half_w = self.track_width / 2
        left = _offset_curve_open(centerline, half_w)
        right = _offset_curve_open(centerline, -half_w)

        if not force and not (_is_simple_polyline(left) and _is_simple_polyline(right)):
            return None

        return centerline, left, right

    def _finalize(self, centerline, left, right):
        self.centerline = centerline
        self.left_boundary = left
        self.right_boundary = right

        # polygone fermé délimitant le ruban de piste (pour affichage et
        # pour le test "point sur la piste") : bord gauche aller, bord droit
        # retour.
        self.ribbon_polygon = np.vstack([left, right[::-1]])

        # segments des deux bords, pré-calculés une fois pour toutes pour le
        # raycasting (les capteurs de la voiture). Piste ouverte -> pas de
        # segment de "fermeture" entre le dernier et le premier point.
        self._wall_segments_a = np.vstack(
            [left[:-1], right[:-1]]
        )  # point A de chaque segment
        self._wall_segments_b = np.vstack(
            [left[1:], right[1:]]
        )  # point B de chaque segment

        tangent_start = centerline[1] - centerline[0]
        tangent_start /= np.linalg.norm(tangent_start)
        # On avance le point de spawn de quelques pixels par rapport à la
        # ligne de départ elle-même : un point pile sur le bord du polygone
        # de la piste donne un test "sur piste" instable (ambigu
        # numériquement), ce qui provoquerait un crash immédiat et injuste
        # au tout premier frame, avant même que le joueur ait pu agir.
        self.start_pos = centerline[0] + tangent_start * 3.0
        self.start_angle = math.atan2(tangent_start[1], tangent_start[0])
        self.start_line = (left[0], right[0])

        tangent_end = centerline[-1] - centerline[-2]
        tangent_end /= np.linalg.norm(tangent_end)
        self.finish_pos = centerline[-1].copy()
        self.finish_angle = math.atan2(tangent_end[1], tangent_end[0])
        self.finish_line = (left[-1], right[-1])
        # point légèrement avant la ligne d'arrivée, utilisé pour spawn une
        # éventuelle voiture "à l'arrivée" sans être pile sur le bord
        self.finish_spawn_pos = centerline[-1] - tangent_end * 3.0

        self._compute_checkpoints(n_checkpoints=14)

    def _compute_checkpoints(self, n_checkpoints=14):
        n = len(self.centerline)
        idxs = np.linspace(0, n - 1, n_checkpoints + 2)[1:-1].astype(int)
        self.checkpoints = [
            (self.left_boundary[i], self.right_boundary[i]) for i in idxs
        ]

    def point_on_track(self, point):
        """Vrai si `point` est à l'intérieur du ruban de piste."""
        return self._point_in_polygon(point, self.ribbon_polygon)

    def distance_to_finish(self, point):
        return float(np.linalg.norm(np.asarray(point) - self.finish_pos))

    @staticmethod
    def _point_in_polygon(point, polygon):
        x, y = point
        inside = False
        n = len(polygon)
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            ):
                inside = not inside
            j = i
        return inside

    def raycast(self, origin, angle, max_distance=200.0):  # pas compris, à regarder
        """Lance un rayon depuis `origin` dans la direction `angle` (radians)
        et renvoie (distance, point_touché) jusqu'au mur le plus proche parmi
        les deux bords de la piste. Si rien n'est touché avant max_distance,
        renvoie (max_distance, point_au_bout_du_rayon).

        Intersection rayon/segment, vectorisée avec numpy pour tester tous
        les segments des murs d'un coup (voir dérivation ci-dessous) plutôt
        que dans une boucle Python, ce qui compte vite quand on a plusieurs
        rayons par voiture et plusieurs voitures par frame.
        """
        O = np.array(origin, dtype=float)
        D = np.array([math.cos(angle), math.sin(angle)])

        A = self._wall_segments_a
        B = self._wall_segments_b

        # Résolution de  O + t*D = A + s*(B-A)  pour t (distance le long du
        # rayon) et s (position le long du segment, doit être dans [0,1]).
        V1 = O - A  # (N,2)
        V2 = B - A  # (N,2)
        V3 = np.array([-D[1], D[0]])  # perpendiculaire à D

        denom = V2 @ V3  # (N,)
        cross_v2_v1 = V2[:, 0] * V1[:, 1] - V2[:, 1] * V1[:, 0]

        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(denom) > 1e-9, cross_v2_v1 / denom, np.inf)
            s = np.where(np.abs(denom) > 1e-9, (V1 @ V3) / denom, np.inf)

        valid = (t >= 0) & (s >= 0) & (s <= 1) & (t <= max_distance)

        if not np.any(valid):
            distance = max_distance
        else:
            distance = max(0.0, float(np.min(t[valid])))

        hit_point = O + D * distance
        return distance, hit_point

    def progress_along_track(self, point):
        """Renvoie un nombre entre 0 (au départ) et 1 (à l'arrivée) indiquant
        à quel point du circuit `point` correspond : l'index du point le
        plus proche sur la centerline, normalisé. Sert de base à la
        récompense en RL : la différence de progression entre deux pas de
        temps donne un signal dense ('on s'est rapproché de l'arrivée de
        combien ?'), bien plus facile à apprendre qu'une récompense qui
        n'arrive qu'à la toute fin.
        """
        p = np.asarray(point, dtype=float)
        dists_sq = np.sum((self.centerline - p) ** 2, axis=1)
        idx = int(np.argmin(dists_sq))
        return idx / (len(self.centerline) - 1)
