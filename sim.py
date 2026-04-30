# simulation/sim.py

import pygame
import sys
import pickle
import random

from env.traffic_env import TrafficEnv
from agent.q_agent import QAgent

# Window
WIDTH, HEIGHT = 700, 700
CENTER = WIDTH // 2

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Traffic RL Simulation 🚦")

clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16)

# Colors
WHITE = (240, 240, 240)
GRAY = (60, 60, 60)
RED = (220, 50, 50)
GREEN = (50, 220, 50)
BLUE = (50, 100, 255)
BLACK = (0, 0, 0)


# 🚗 Car Class
class Car:
    SIZE = 12
    GAP = 22

    def __init__(self, direction):
        self.direction = direction
        self.speed = 1.2

        if direction == "N":
            self.x = CENTER - 20
            self.y = -40
        elif direction == "S":
            self.x = CENTER + 10
            self.y = HEIGHT + 40
        elif direction == "E":
            self.x = WIDTH + 40
            self.y = CENTER - 20
        elif direction == "W":
            self.x = -40
            self.y = CENTER + 10

    def blocked_by_car(self, cars):
        for other in cars:
            if other == self or other.direction != self.direction:
                continue

            if self.direction == "N" and 0 < other.y - self.y < self.GAP:
                return True
            if self.direction == "S" and 0 < self.y - other.y < self.GAP:
                return True
            if self.direction == "E" and 0 < self.x - other.x < self.GAP:
                return True
            if self.direction == "W" and 0 < other.x - self.x < self.GAP:
                return True

        return False

    def blocked_by_signal(self, action):
        stop_line = 90

        if self.direction in ["N", "S"] and action != 0:
            if (self.direction == "N" and self.y >= CENTER - stop_line) or \
               (self.direction == "S" and self.y <= CENTER + stop_line):
                return True

        if self.direction in ["E", "W"] and action != 1:
            if (self.direction == "E" and self.x <= CENTER + stop_line) or \
               (self.direction == "W" and self.x >= CENTER - stop_line):
                return True

        return False

    def move(self, action, cars):
        if self.blocked_by_car(cars):
            return
        if self.blocked_by_signal(action):
            return

        if self.direction == "N":
            self.y += self.speed
        elif self.direction == "S":
            self.y -= self.speed
        elif self.direction == "E":
            self.x -= self.speed
        elif self.direction == "W":
            self.x += self.speed

    def draw(self):
        pygame.draw.rect(
            screen, BLUE,
            (int(self.x), int(self.y), self.SIZE, self.SIZE),
            border_radius=3
        )


# 📦 Load trained model
def load_q_table(agent):
    try:
        with open("results/q_table.pkl", "rb") as f:
            agent.q_table = pickle.load(f)
        print("Q-table loaded ✅")
    except:
        print("No trained model found ❌")


# 🚨 Collision check
def check_collision(cars):
    for i in range(len(cars)):
        for j in range(i + 1, len(cars)):
            c1, c2 = cars[i], cars[j]

            if abs(c1.x - c2.x) < 10 and abs(c1.y - c2.y) < 10:
                return True
    return False


# 🛣 Draw roads
def draw_roads():
    screen.fill(WHITE)
    pygame.draw.rect(screen, GRAY, (CENTER - 50, 0, 100, HEIGHT))
    pygame.draw.rect(screen, GRAY, (0, CENTER - 50, WIDTH, 100))


# 🚦 Lights
def draw_lights(action):
    if action == 0:
        pygame.draw.circle(screen, GREEN, (CENTER, CENTER - 70), 12)
        pygame.draw.circle(screen, RED, (CENTER + 70, CENTER), 12)
    else:
        pygame.draw.circle(screen, RED, (CENTER, CENTER - 70), 12)
        pygame.draw.circle(screen, GREEN, (CENTER + 70, CENTER), 12)


# 📊 Info panel
def draw_info(state, action, agent):
    text1 = f"Action: {'NS Green' if action==0 else 'EW Green'}"
    text2 = f"State: {state}"

    state_t = agent._state_to_tuple(state)
    q0 = agent.q_table.get((state_t, 0), 0)
    q1 = agent.q_table.get((state_t, 1), 0)

    text3 = f"Q(NS): {round(q0,2)}  Q(EW): {round(q1,2)}"

    screen.blit(font.render(text1, True, BLACK), (10, 10))
    screen.blit(font.render(text2, True, BLACK), (10, 30))
    screen.blit(font.render(text3, True, BLACK), (10, 50))


# 🚨 Popup
def draw_collision_popup(timer):
    if timer > 0:
        text = font.render("⚠️ COLLISION! PENALTY -50", True, RED)
        screen.blit(text, (220, 120))


# 🎮 MAIN
def main():
    env = TrafficEnv()
    agent = QAgent()

    load_q_table(agent)

    state = env.reset()
    cars = []

    action = 0
    timer = 0
    SWITCH_TIME = 60

    collision_timer = 0

    running = True
    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # signal timing
        if timer == 0:
            action = agent.choose_action(state)

        timer += 1
        if timer >= SWITCH_TIME:
            timer = 0

        next_state, reward = env.step(action)
        state = next_state

        # 🚗 limit cars per lane
        lane_counts = {"N":0, "S":0, "E":0, "W":0}
        for car in cars:
            lane_counts[car.direction] += 1

        direction = random.choice(["N","S","E","W"])
        if lane_counts[direction] < 4 and random.random() < 0.2:
            cars.append(Car(direction))

        # move cars
        for car in cars:
            car.move(action, cars)

        # collision detection
        if check_collision(cars):
            collision_timer = 60

        if collision_timer > 0:
            collision_timer -= 1

        # remove off-screen
        cars = [
            car for car in cars
            if -60 < car.x < WIDTH+60 and -60 < car.y < HEIGHT+60
        ]

        draw_roads()
        draw_lights(action)

        for car in cars:
            car.draw()

        draw_info(state, action, agent)
        draw_collision_popup(collision_timer)

        pygame.display.flip()


if __name__ == "__main__":
    main()