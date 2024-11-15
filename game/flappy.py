import numpy as np
import sys
import random
import pygame
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

# Constants for game speed and display dimensions
FRAME_RATE = 30
DISPLAY_WIDTH = 288
DISPLAY_HEIGHT = 512

# Pygame initialization and display setup
pygame.init()
FRAME_CLOCK = pygame.time.Clock()
DISPLAY_SURFACE = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption('Flappy Bird Clone')


def load_assets():
    """Loads and returns game assets including sprites and hitmasks."""
    # Player bird sprite paths with animation frames
    PLAYER_FRAMES = (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png'
    )

    # Background image path
    BG_IMAGE_PATH = 'assets/sprites/background-black.png'

    # Pipe image path
    PIPE_IMAGE_PATH = 'assets/sprites/pipe-green.png'

    assets, hit_masks = {}, {}

    # Number sprites for score display
    assets['numbers'] = [
        pygame.image.load(f'assets/sprites/{i}.png').convert_alpha() for i in range(10)
    ]

    # Ground/base sprite
    assets['ground'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # Background sprite
    assets['background'] = pygame.image.load(BG_IMAGE_PATH).convert()

    # Player sprites for animation
    assets['player'] = [
        pygame.image.load(frame).convert_alpha() for frame in PLAYER_FRAMES
    ]

    # Pipe sprites for upper and lower pipes
    assets['pipe'] = (
        pygame.transform.rotate(pygame.image.load(PIPE_IMAGE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_IMAGE_PATH).convert_alpha(),
    )

    # Hitmasks for collision detection for pipes and player
    hit_masks['pipe'] = (generate_hitmask(assets['pipe'][0]), generate_hitmask(assets['pipe'][1]))
    hit_masks['player'] = [generate_hitmask(frame) for frame in assets['player']]

    return assets, hit_masks

def generate_hitmask(sprite):
    """Generates a hitmask for a sprite based on its alpha channel."""
    mask = [[bool(sprite.get_at((x, y))[3]) for y in range(sprite.get_height())] for x in range(sprite.get_width())]
    return mask

ASSETS, HIT_MASKS = load_assets()
PIPE_GAP = 100  # Gap size between upper and lower pipes
GROUND_Y = DISPLAY_HEIGHT * 0.79

# Dimensions of the player, pipe, and background sprites
PLAYER_WIDTH = ASSETS['player'][0].get_width()
PLAYER_HEIGHT = ASSETS['player'][0].get_height()
PIPE_WIDTH = ASSETS['pipe'][0].get_width()
PIPE_HEIGHT = ASSETS['pipe'][0].get_height()
BACKGROUND_WIDTH = ASSETS['background'].get_width()

PLAYER_ANIMATION_CYCLE = cycle([0, 1, 2, 1])


class FlappyBirdGame:
    def __init__(self):
        self.score = self.player_frame = self.loop_count = 0
        self.player_x = int(DISPLAY_WIDTH * 0.2)
        self.player_y = int((DISPLAY_HEIGHT - PLAYER_HEIGHT) / 2)
        self.ground_x = 0
        self.ground_scroll_speed = ASSETS['ground'].get_width() - BACKGROUND_WIDTH

        pipe1 = create_pipe()
        pipe2 = create_pipe()
        self.upper_pipes = [
            {'x': DISPLAY_WIDTH, 'y': pipe1[0]['y']},
            {'x': DISPLAY_WIDTH + (DISPLAY_WIDTH / 2), 'y': pipe2[0]['y']},
        ]
        self.lower_pipes = [
            {'x': DISPLAY_WIDTH, 'y': pipe1[1]['y']},
            {'x': DISPLAY_WIDTH + (DISPLAY_WIDTH / 2), 'y': pipe2[1]['y']},
        ]

        # Player movement parameters
        self.pipe_scroll_x = -4
        self.player_vel_y = 0  # Player's current vertical speed
        self.player_max_speed_down = 10  # Max falling speed
        self.player_max_speed_up = -8  # Max rising speed
        self.player_gravity = 1  # Gravity effect on player
        self.flap_boost = -9  # Vertical boost on flap
        self.has_flapped = False  # Flap state

    def update_frame(self, actions):
        pygame.event.pump()

        reward = 0.1
        game_over = False

        if sum(actions) != 1:
            raise ValueError('Only one action can be selected!')

        # Action: flap the bird if actions[1] is selected
        if actions[1] == 1 and self.player_y > -2 * PLAYER_HEIGHT:
            self.player_vel_y = self.flap_boost
            self.has_flapped = True

        # Update score when passing pipes
        player_mid_pos = self.player_x + PLAYER_WIDTH / 2
        for pipe in self.upper_pipes:
            pipe_mid_pos = pipe['x'] + PIPE_WIDTH / 2
            if pipe_mid_pos <= player_mid_pos < pipe_mid_pos + 4:
                self.score += 1
                reward = 1

        # Animation and ground scrolling update
        if (self.loop_count + 1) % 3 == 0:
            self.player_frame = next(PLAYER_ANIMATION_CYCLE)
        self.loop_count = (self.loop_count + 1) % 30
        self.ground_x = -((-self.ground_x + 100) % self.ground_scroll_speed)

        # Player movement and gravity application
        if self.player_vel_y < self.player_max_speed_down and not self.has_flapped:
            self.player_vel_y += self.player_gravity
        if self.has_flapped:
            self.has_flapped = False
        self.player_y += min(self.player_vel_y, GROUND_Y - self.player_y - PLAYER_HEIGHT)
        if self.player_y < 0:
            self.player_y = 0

        # Move pipes leftward
        for u_pipe, l_pipe in zip(self.upper_pipes, self.lower_pipes):
            u_pipe['x'] += self.pipe_scroll_x
            l_pipe['x'] += self.pipe_scroll_x

        # Add new pipe when the first pipe is near the left edge of the screen
        if 0 < self.upper_pipes[0]['x'] < 5:
            new_pipe = create_pipe()
            self.upper_pipes.append(new_pipe[0])
            self.lower_pipes.append(new_pipe[1])

        # Remove off-screen pipes
        if self.upper_pipes[0]['x'] < -PIPE_WIDTH:
            self.upper_pipes.pop(0)
            self.lower_pipes.pop(0)

        # Check for collision
        collision = check_collision({'x': self.player_x, 'y': self.player_y, 'index': self.player_frame}, self.upper_pipes, self.lower_pipes)
        if collision:
            game_over = True
            self.__init__()
            reward = -1

        # Draw game objects
        DISPLAY_SURFACE.blit(ASSETS['background'], (0, 0))
        for u_pipe, l_pipe in zip(self.upper_pipes, self.lower_pipes):
            DISPLAY_SURFACE.blit(ASSETS['pipe'][0], (u_pipe['x'], u_pipe['y']))
            DISPLAY_SURFACE.blit(ASSETS['pipe'][1], (l_pipe['x'], l_pipe['y']))

        DISPLAY_SURFACE.blit(ASSETS['ground'], (self.ground_x, GROUND_Y))
        DISPLAY_SURFACE.blit(ASSETS['player'][self.player_frame], (self.player_x, self.player_y))

        screen_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FRAME_CLOCK.tick(FRAME_RATE)
        
        return screen_data, reward, game_over

def create_pipe():
    """Generates a pair of pipes with a gap at a random vertical position."""
    gap_positions = [20, 30, 40, 50, 60, 70, 80, 90]
    gap_y = random.choice(gap_positions) + int(GROUND_Y * 0.2)
    pipe_x = DISPLAY_WIDTH + 10

    return [
        {'x': pipe_x, 'y': gap_y - PIPE_HEIGHT},  # upper pipe
        {'x': pipe_x, 'y': gap_y + PIPE_GAP},  # lower pipe
    ]


def render_score(score):
    """Renders the player's score in the center of the screen."""
    score_digits = [int(d) for d in str(score)]
    total_width = sum(ASSETS['numbers'][d].get_width() for d in score_digits)
    x_offset = (DISPLAY_WIDTH - total_width) / 2

    for digit in score_digits:
        DISPLAY_SURFACE.blit(ASSETS['numbers'][digit], (x_offset, DISPLAY_HEIGHT * 0.1))
        x_offset += ASSETS['numbers'][digit].get_width()


def check_collision(player, upper_pipes, lower_pipes):
    """Detects if the player collides with the ground or pipes."""
    player_index = player['index']
    player_rect = pygame.Rect(player['x'], player['y'], PLAYER_WIDTH, PLAYER_HEIGHT)

    if player_rect.bottom >= GROUND_Y - 1:
        return True

    for u_pipe, l_pipe in zip(upper_pipes, lower_pipes):
        u_pipe_rect = pygame.Rect(u_pipe['x'], u_pipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
        l_pipe_rect = pygame.Rect(l_pipe['x'], l_pipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

        player_hitmask = HIT_MASKS['player'][player_index]
        u_pipe_hitmask = HIT_MASKS['pipe'][0]
        l_pipe_hitmask = HIT_MASKS['pipe'][1]

        u_collision = pixel_collision(player_rect, u_pipe_rect, player_hitmask, u_pipe_hitmask)
        l_collision = pixel_collision(player_rect, l_pipe_rect, player_hitmask, l_pipe_hitmask)

        if u_collision or l_collision:
            return True

    return False

def pixel_collision(rect1, rect2, hitmask1, hitmask2):
    """Checks pixel-level collision between two objects."""
    rect_intersection = rect1.clip(rect2)

    if rect_intersection.width == 0 or rect_intersection.height == 0:
        return False

    x1, y1 = rect_intersection.x - rect1.x, rect_intersection.y - rect1.y
    x2, y2 = rect_intersection.x - rect2.x, rect_intersection.y - rect2.y

    for x in range(rect_intersection.width):
        for y in range(rect_intersection.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False
