from track_generator import Track
from car import Car
import random
from constante import ACTIONS, SPEED_BINS, SENSOR_BINS, DT


def discretize_state(sensor_readings, speed, sensor_max, speed_max):
    """Transforme les mesures continues (distances des capteurs + vitesse)
    en un tuple d'entiers -- c'est ce tuple qui sert de clé dans la Q-table.
    """
    sensor_disc = tuple(
        min(SENSOR_BINS - 1, int((r / sensor_max) * SENSOR_BINS))
        for r in sensor_readings
    )
    # la vitesse peut être négative (marche arrière) jusqu'à -speed_max/2
    speed_norm = (speed + speed_max / 2) / (speed_max * 1.5)
    speed_disc = min(
        SPEED_BINS - 1, max(0, int(speed_norm * SPEED_BINS))
    )  # vitesse negative = 0 ?? voir si correct
    return sensor_disc + (
        speed_disc,
    )  # ajout de la vitesse dans le tupple des distances ?


class Emulate:
    def __init__(
        self, width=900, height=700, track_width=70, fixed_seed=None, max_steps=600
    ):
        self.width = width
        self.height = height
        self.track_width = track_width
        self.fixed_seed = fixed_seed
        self.max_steps = max_steps
        self.n_actions = len(ACTIONS)

        self.track = None
        self.car = None
        self.steps = 0
        self.prev_progress = 0.0

    def reset(self):
        if self.fixed_seed is not None:
            # même circuit à chaque épisode : on ne le génère qu'une fois
            # (Track est immuable après génération, donc réutilisable tel quel)
            if self.track is None:
                self.track = Track(
                    self.width, self.height, self.track_width, seed=self.fixed_seed
                )
        else:
            self.track = Track(
                self.width,
                self.height,
                self.track_width,
                seed=random.randint(0, 999_999),
            )
        self.car = Car(*self.track.start_pos, self.track.start_angle)
        self.steps = 0

        self.prev_progress = self.track.progress_along_track(self.track.start_pos)
        return self._get_state()

    def step(self, action_index):
        throttle, steer = ACTIONS[action_index]
        self.car.update(DT, (throttle, steer))
        self.steps += 1

        reward = -0.01  # petite pénalité de temps : incite à finir vite
        done = False
        info = {}

        if not self.track.point_on_track((self.car.x, self.car.y)):
            self.car.crash()

        if self.car.crashed:
            reward -= 100.0
            done = True
            info["result"] = "crash"
        else:
            # récompense de progression : positive si on avance vers
            # l'arrivée, négative si on recule -- signal dense à chaque pas,
            # plutôt que de n'être récompensé qu'à l'arrivée finale.
            progress = self.track.progress_along_track((self.car.x, self.car.y))
            reward += (
                progress - self.prev_progress
            ) * 100.0  # * 100 ? verifier si ya mieux
            self.prev_progress = progress

            if (
                self.track.distance_to_finish((self.car.x, self.car.y))
                < self.track_width / 2
            ):
                self.car.finish()
                reward += 50.0
                done = True
                info["result"] = "finish"

        if self.steps >= self.max_steps and not done:
            done = True
            info["result"] = "timeout"

        return self._get_state(), reward, done, info

    #
    # def train():
    #
    #
    # def evaluate():
    #
    #
    # def play():

    def _get_state(self):
        distances = self.car.get_sensor_distances(self.track)
        return discretize_state(
            distances, self.car.speed, self.car.SENSOR_MAX_DISTANCE, self.car.max_speed
        )
