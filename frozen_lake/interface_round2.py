### Main function to learn the POMCP policy for Mastermind (Simulation)
# Implemented by Manisha Natarajan
# Last Update: 04/24/2023

import os

from frozen_lake.frozen_lake_env import *
from frozen_lake.solver import *
from frozen_lake.root_node import *
from frozen_lake.robot_action_node import *
from frozen_lake.human_action_node import *
from frozen_lake.simulated_human import *
from utils import *
import time
import pygame
import string
import json

order = [4, 8, 6]
heuristic_order = [0, 1]  # First one is the order of interrupting agent, second is the order of taking control agent.
random.shuffle(heuristic_order)
CONDITION = {
    'practice': [0, 1, 2, 3],
    'pomcp': [order[0], order[0] + 1],
    'pomcp_inverse': [order[1], order[1] + 1],
    'interrupt': [order[2] + heuristic_order[0]],
    'take_control': [order[2] + heuristic_order[1]]
}
# order = [4, 5, 9, 8, 7, 6]
expOrder = [0, 1, 2, 3, order[0], order[0] + 1, order[1], order[1] + 1, order[2] + heuristic_order[0], order[2] + heuristic_order[1]]
print("expOrder", expOrder)


mapOrder = [4, 5, 7, 10, 11, 12]
random.shuffle(mapOrder)
# mapOrder = [10, 10, 10, 10, 10, 10]
mapOrder = [0, 2, 3, 6] + mapOrder
print("mapOrder", mapOrder)

username = ''.join(
    random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(5))
filename = 'frozen_lake/files/user_study/' + username + ".json"
userdata = {}
userdata['expOrder'] = expOrder
userdata['mapOrder'] = mapOrder


class HeuristicAgent:
    def __init__(self, type):
        self.robot_action = 0
        self.type = type
        self.num_interrupt = 0  # Number of interruption when taking a longer path (<3)

    def get_action(self, env):
        position = env.world_state[0]
        last_position = env.world_state[1]
        robot_slippery = env.world_state[3]
        robot_err = env.world_state[5]
        last_path = env.find_shortest_path(env.desc, robot_slippery, last_position, env.ncol)
        current_path = env.find_shortest_path(env.desc, robot_slippery, position, env.ncol)
        if env.desc[position // env.ncol, position % env.ncol] in b'HS' and \
                ((position // env.ncol, position % env.ncol) not in env.robot_err or
                 (position // env.ncol, position % env.ncol) in robot_err):
            self.robot_action = self.type
        elif env.desc[position // env.ncol, position % env.ncol] in b'F' and \
                ((position // env.ncol, position % env.ncol) in env.robot_err and
                 (position // env.ncol, position % env.ncol) not in robot_err):
            self.robot_action = self.type
        elif len(current_path) > 1 and len(last_path) > 1 and len(last_path) <= len(current_path) and \
                self.num_interrupt < 3:
            self.robot_action = self.type
            self.num_interrupt += 1
        else:
            self.robot_action = 0
        return self.robot_action


class FrozenLakeEnvInterface(FrozenLakeEnv):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def render(self, round_num, human_action, robot_action, world_state, end_detecting=0, truncated=False,
               timeout=False):
        window_width = self.window_size[0]
        window_height = self.window_size[1]
        map_size = self.window_size[0] - 256*2
        if robot_action:
            robot_type, robot_direction = robot_action
        else:
            robot_type, robot_direction = None, None
        if human_action:
            _, detecting, human_direction = human_action
        else:
            detecting, human_direction = None, None
        position, last_position, human_slippery, robot_slippery, human_err, robot_err = world_state

        class TextBox(pygame.sprite.Sprite):
            def __init__(self, surface):
                pygame.sprite.Sprite.__init__(self)
                self.initFont()
                self.initImage(surface)
                self.initGroup()

            def initFont(self):
                pygame.font.init()
                self.font = pygame.font.Font(pygame.font.match_font('calibri'), 20)
                self.font_bold = pygame.font.Font(pygame.font.match_font('calibri', bold=True), 20)

            def initImage(self, surface):
                # self.image = pygame.Surface((200, 80))
                self.image = surface
                self.ncol = 8
                self.system_rect = pygame.Rect(0, 0, 256, map_size)
                self.robot_rect = pygame.Rect(256 + map_size, 0, 256, map_size)
                self.image.fill((255, 255, 255), rect=self.system_rect)
                self.image.fill((255, 255, 255), rect=self.robot_rect)
                self.system_top = map_size / 2
                self.robot_top = 100
                self.robot_left = map_size + 256
                self.system_left = map_size + 256

            def setText(self, robot=None, system=None):
                tmp = pygame.display.get_surface()

                if system is not None:
                    x_pos = self.system_left + 5
                    y_pos = self.system_top + 5
                    x = self.font_bold.render("System:", True, (0, 0, 0))
                    tmp.blit(x, (x_pos, y_pos))
                    y_pos += 20
                    words = system.split(' ')
                    for t in words:
                        # print(t, x_pos)
                        if t == 'NOT':
                            x = self.font.render(t + " ", True, (255, 0, 0))
                        elif t in ["ENTER", "SPACE", "BACKSPACE"]:
                            x = self.font_bold.render(t + " ", True, (0, 0, 0))
                        else:
                            x = self.font.render(t + " ", True, (0, 0, 0))
                        if x_pos + x.get_width() < self.image.get_width() - 5:
                            tmp.blit(x, (x_pos, y_pos))
                            x_pos += x.get_width()
                        else:
                            x_pos = self.system_left + 5
                            y_pos += 20
                            tmp.blit(x, (x_pos, y_pos))
                            x_pos += x.get_width()
                pygame.draw.line(tmp, (0, 0, 0), (self.robot_left, self.system_top),
                                 (self.robot_left + 256, self.system_top))

                if robot is not None:
                    x_pos = self.robot_left + 5
                    y_pos = self.robot_top + 5
                    x = self.font_bold.render("Robot:", True, (0, 0, 0))
                    tmp.blit(x, (x_pos, y_pos))
                    y_pos += 20
                    words = robot.split(' ')
                    for t in words:
                        if t == 'n':
                            x = self.font.render("", True, (0, 0, 0))
                            x_pos = self.robot_left + 5
                            y_pos += 36
                        elif t not in ["slippery", "region.", "hole.", "longer", "way."]:
                            x = self.font.render(t + " ", True, (0, 0, 0))
                        else:
                            x = self.font_bold.render(t + " ", True, (0, 0, 0))
                        if x_pos + x.get_width() < self.image.get_width() - 5:
                            tmp.blit(x, (x_pos, y_pos))
                            x_pos += x.get_width()
                        else:
                            x_pos = self.robot_left + 5
                            y_pos += 20
                            tmp.blit(x, (x_pos, y_pos))
                            x_pos += x.get_width()

            def initGroup(self):
                self.group = pygame.sprite.GroupSingle()
                self.group.add(self)

        if self.window_surface is None:
            pygame.init()

            pygame.display.init()
            pygame.display.set_caption("Frozen Lake")
            self.window_surface = pygame.display.set_mode(self.window_size)

        assert (
                self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = os.path.join(os.path.dirname(__file__), "img/hole_new.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.cracked_hole_img is None:
            file_name = os.path.join(os.path.dirname(__file__), "img/cracked_hole_1.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        # if self.ice_img is None:
        #     file_name = os.path.join(os.path.dirname(__file__), "img/ice.png")
        #     self.ice_img = pygame.transform.scale(
        #         pygame.image.load(file_name), self.cell_size
        #     )
        if self.goal_img is None:
            file_name = os.path.join(os.path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = os.path.join(os.path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.elf_images is None:
            elfs = [
                os.path.join(os.path.dirname(__file__), "img/robot{}.png".format(int(round_num / 2))),
                os.path.join(os.path.dirname(__file__), "img/robot{}.png".format(int(round_num / 2))),
                os.path.join(os.path.dirname(__file__), "img/robot{}.png".format(int(round_num / 2))),
                os.path.join(os.path.dirname(__file__), "img/robot{}.png".format(int(round_num / 2))),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]

        if self.fog_img is None:
            file_name = os.path.join(os.path.dirname(__file__), "img/ice.png")
            self.fog_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        if self.smoke_img is None:
            file_name = os.path.join(os.path.dirname(__file__), "img/steam_2.png")
            self.smoke_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        if self.slippery_img is None:
            file_name = os.path.join(os.path.dirname(__file__), "img/slippery_1.png")
            self.slippery_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        desc = self.desc.tolist()
        # human_map = self.human_map.tolist()
        foggy = self.fog.tolist()
        # assert isinstance(human_map, list), f"human_map should be a list or an array, got {human_map}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0] + 256, y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                # self.window_surface.blit(self.ice_img, pos)
                if foggy[y][x] == b"F":
                    # self.window_surface.fill((255, 255, 255), rect=rect)
                    self.window_surface.blit(self.fog_img, pos)
                else:
                    # self.window_surface.blit(self.ice_img, pos)
                    self.window_surface.fill((255, 255, 255), rect=rect)
                if (y, x) in human_slippery:
                    self.window_surface.blit(self.slippery_img, pos)
                    if foggy[y][x] == b"F":
                        self.window_surface.blit(self.smoke_img, pos)
                # if desc[y][x] == b"S":
                #     self.window_surface.blit(self.slippery_img, pos)
                # if (desc[y][x] == b"S" and (y, x) not in self.robot_err) or (
                #         desc[y][x] not in b"S" and (y, x) in self.robot_err):
                #     self.window_surface.blit(self.slippery_img, pos)
                if desc[y][x] == b"H":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"B":
                    self.window_surface.blit(self.start_img, pos)
                # if self.robot_map[y][x] == b"H":
                #     self.window_surface.blit(self.goal_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)


        # Add legend
        self.window_surface.fill((255, 255, 255), rect=(0, map_size, window_width, window_height-map_size))
        pygame.draw.line(self.window_surface, (0, 0, 0), (0, map_size),
                         (window_width, map_size))
        pygame.font.init()
        font_bold = pygame.font.Font(pygame.font.match_font('calibri', bold=True), 25)
        x = font_bold.render("Legend", True, (0, 0, 0))
        self.window_surface.blit(x, (10, map_size+20))
        # Ice
        left_pos = 10 + x.get_width() + 20
        top_pos = map_size + 20
        font = pygame.font.Font(pygame.font.match_font('calibri'), 25)
        x = font.render("Non-fog", True, (0, 0, 0))
        self.window_surface.blit(x, (left_pos, top_pos))
        self.window_surface.blit(self.smoke_img, (left_pos + x.get_width() + 5, top_pos))
        pygame.draw.rect(self.window_surface, (180, 200, 230), ((left_pos + x.get_width() + 5, top_pos), self.cell_size), 1)
        # Fog
        font = pygame.font.Font(pygame.font.match_font('calibri'), 25)
        x = font.render("       Fog", True, (0, 0, 0))
        self.window_surface.blit(x, (left_pos, top_pos + 120))
        self.window_surface.blit(self.fog_img, (left_pos + x.get_width() + 5, top_pos + 120))
        pygame.draw.rect(self.window_surface, (180, 200, 230), ((left_pos + x.get_width() + 5, top_pos + 120), self.cell_size), 1)
        # Slippery
        left_pos += 120 + x.get_width()
        font = pygame.font.Font(pygame.font.match_font('calibri'), 25)
        x = font.render("Slippery", True, (0, 0, 0))
        self.window_surface.blit(x, (left_pos, top_pos))
        x_p = font.render("(No fog)", True, (0, 0, 0))
        self.window_surface.blit(x_p, (left_pos, top_pos + x.get_height()))
        self.window_surface.blit(self.slippery_img, (left_pos + x.get_width() + 5, top_pos))
        pygame.draw.rect(self.window_surface, (180, 200, 230), ((left_pos + x.get_width() + 5, top_pos), self.cell_size), 1)

        x = font.render("Slippery", True, (0, 0, 0))
        self.window_surface.blit(x, (left_pos, top_pos + 120))
        x_p = font.render("(Fog)", True, (0, 0, 0))
        self.window_surface.blit(x_p, (left_pos, top_pos + x.get_height() + 120))
        self.window_surface.blit(self.slippery_img, (left_pos + x.get_width() + 5, top_pos + 120))
        self.window_surface.blit(self.smoke_img, (left_pos + x.get_width() + 5, top_pos + 120))
        pygame.draw.rect(self.window_surface, (180, 200, 230), ((left_pos + x.get_width() + 5, top_pos+120), self.cell_size), 1)
        # Hole
        left_pos += 120 + x.get_width()
        font = pygame.font.Font(pygame.font.match_font('calibri'), 25)
        x = font.render("Hole", True, (0, 0, 0))
        self.window_surface.blit(x, (left_pos, top_pos))
        self.window_surface.blit(self.hole_img, (left_pos + x.get_width() + 5, top_pos))
        pygame.draw.rect(self.window_surface, (180, 200, 230), ((left_pos + x.get_width() + 5, top_pos), self.cell_size), 1)
        # Goal
        left_pos += 120 + x.get_width()
        font = pygame.font.Font(pygame.font.match_font('calibri'), 25)
        x = font.render("Goal", True, (0, 0, 0))
        self.window_surface.blit(x, (left_pos, top_pos))
        self.window_surface.blit(self.goal_img, (left_pos + x.get_width() + 5, top_pos))

        # paint the elf
        bot_row, bot_col = position // self.ncol, position % self.ncol
        cell_rect = (bot_col * self.cell_size[0] + 256, bot_row * self.cell_size[1])
        last_action = human_direction
        elf_img = self.elf_images[0]

        # Robot notification
        textbox = TextBox(self.window_surface)
        ACTIONS = ["LEFT", "DOWN", "RIGHT", "UP"]
        human_action_name = None
        robot_action_name = None
        if last_action != None:
            human_action_name = ACTIONS[last_action]
        if robot_direction != None:
            robot_action_name = ACTIONS[robot_direction]
            # self.robot_action = None
        if timeout:
            # self.window_surface.blit(elf_img, cell_rect)
            textbox.setText(
                system="You've run out of the step number. You failed. Please finish the survey and ask the experimenter to start a new game.")
        elif detecting and human_direction is None:
            self.window_surface.blit(elf_img, cell_rect)
            textbox.setText(system="You're entering detection mode. Press arrow keys to check the surrounding grids.")
        elif end_detecting == 1:
            self.window_surface.blit(elf_img, cell_rect)
            textbox.setText(system="You're exiting detection mode and back to navigation mode.")
        elif end_detecting == 2:
            self.window_surface.blit(elf_img, cell_rect)
            textbox.setText(system="You're out of attempts for using the detection sensor. Press BACKSPACE again to exit detection mode.")
        elif detecting:
            self.window_surface.blit(elf_img, cell_rect)
            s = self.move(position, human_direction)
            # print(position, human_direction, s // self.ncol, s % self.ncol)
            if desc[s // self.ncol][s % self.ncol] in b'SH':
                is_slippery = True
            else:
                is_slippery = False
            left = (s % self.ncol) * self.cell_size[0] + 256
            top = (s // self.ncol) * self.cell_size[1]
            if is_slippery:
                pygame.draw.rect(self.window_surface, (255, 0, 0),
                                 pygame.Rect(left, top, self.cell_size[0], self.cell_size[1]), 4)
                # pygame.display.flip()
                textbox.setText(
                    system="The region you're detecting is NOT safe! Press BACKSPACE again to exit detection mode.")
            else:
                pygame.draw.rect(self.window_surface, (0, 255, 0),
                                 pygame.Rect(left, top, self.cell_size[0], self.cell_size[1]), 4)
                # pygame.display.flip()
                textbox.setText(
                    system="The region you're detecting is safe. Press BACKSPACE again to exit detection mode.")
        else:
            if robot_type == 1:  # interrupt
                system_prompt = "Press ENTER and then make your next choice."
                robot_prompt = "Your last choice was {}. I chose to stay. Please choose an action again.".format(
                    human_action_name)
            elif robot_type == 2:  # control
                system_prompt = "Press ENTER and then make your next choice."
                robot_prompt = "Your last choice was {}. My action was {}.".format(
                    human_action_name, robot_action_name)
            elif robot_type == 3:  # interrupt_w_explain
                last_row = last_position // env.ncol
                last_col = last_position % env.ncol
                if (desc[last_row][last_col] in b'S' and (
                        (last_row, last_col) not in self.robot_err or ((last_row, last_col) in robot_err))) or \
                        (desc[last_row][last_col] in b'F' and (
                                (last_row, last_col) in self.robot_err and ((last_row, last_col) not in robot_err))):
                    system_prompt = "Press ENTER and then make your next choice."
                    robot_prompt = "Your last choice was {}. I chose to stay. n Going {} might step into a slippery region. " \
                                   "Please choose an action again. ".format(
                        human_action_name, human_action_name)
                elif desc[last_row][last_col] in b'H':
                    system_prompt = "Press ENTER and then make your next choice."
                    robot_prompt = "Your last choice was {}. I chose to stay. n Going {} will step into a hole. " \
                                   "Please choose an action again. ".format(
                        human_action_name, human_action_name)
                else:
                    system_prompt = "Press ENTER and then make your next choice."
                    robot_prompt = "Your last choice was {}. I chose to stay. n Going {} might take a longer way. " \
                                   "Please choose an action again.".format(
                        human_action_name, human_action_name)
            elif robot_type == 4:  # control_w_explain
                last_row = last_position // env.ncol
                last_col = last_position % env.ncol
                if (desc[last_row][last_col] in b'S' and ((last_row, last_col) not in self.robot_err or ((last_row, last_col) in robot_err))) or \
                    (desc[last_row][last_col] in b'F' and ((last_row, last_col) in self.robot_err and ((last_row, last_col) not in robot_err))):
                    system_prompt = "Press ENTER and then make your next choice."
                    robot_prompt = "Robot: Your last choice was {}. My action was {}. n Going {} might step into a slippery region. ".format(
                        human_action_name, robot_action_name, human_action_name)
                elif desc[last_row][last_col] in b'H':
                    system_prompt = "Press ENTER and then make your next choice."
                    robot_prompt = "Your last choice was {}. My action was {}. n Going {} will step into a hole. ".format(
                        human_action_name, robot_action_name, human_action_name)
                else:
                    system_prompt = "Press ENTER and then make your next choice."
                    robot_prompt = "Your last choice was {}. My action was {}. n Going {} might take a longer way. ".format(
                        human_action_name, robot_action_name, human_action_name)
            else:
                robot_prompt = "Your last choice was {}. I followed your choice.".format(human_action_name)

            # if desc[bot_row][bot_col] == b"SH":
            if truncated:
                last_row, last_col = last_position // self.ncol, last_position % self.ncol
                if robot_type == 0:
                    hole_row, hole_col = last_row, last_col
                else:
                    hole_row, hole_col = self.inc(last_row, last_col, robot_direction)
                last_cell_rect = (hole_col * self.cell_size[0] + 256, hole_row * self.cell_size[1])
                if desc[hole_row][hole_col] in b"H":
                    last_cell_rect = (hole_col * self.cell_size[0] + 256, hole_row * self.cell_size[1])
                    print(hole_row, hole_col)
                else:
                    for r in [-1, 0, 1]:
                        for c in [-1, 0, 1]:
                            hole_row = last_row + r
                            hole_col = last_col + c
                            if 0 <= hole_row < self.ncol and 0 <= hole_col < self.ncol:
                                if desc[hole_row][hole_col] in b'H':
                                    last_cell_rect = (hole_col * self.cell_size[0] + 256, hole_row * self.cell_size[1])
                                    print(hole_row, hole_col)
                                    break
                self.window_surface.blit(self.cracked_hole_img, last_cell_rect)
                if round_num in CONDITION['practice']:
                    system_prompt = "You failed. Please press ENTER to restart."
                    robot_prompt = "Your last choice was {}. I followed your choice. I slipped into a hole.".format(
                        human_action_name,
                        robot_action_name)
                    # textbox.setText(
                    #     "Robot: Your last choice was {}. My action was {}. I slipped into a hole. We failed.Please press ENTER to restart.".format(
                    #         action, action))
                    textbox.setText(robot=robot_prompt, system=system_prompt)
                    self.window_surface.blit(self.elf_images[0], (256 + map_size, 0))
                elif robot_type in [2, 4]: # taking control
                    system_prompt = "You failed. Please press ENTER to restart."
                    robot_prompt = "Your last choice was {}. My action was {}. I slipped into a hole.".format(
                        human_action_name, robot_action_name)
                    textbox.setText(robot=robot_prompt, system=system_prompt)
                    self.window_surface.blit(self.elf_images[0], (256 + map_size, 0))
                    # textbox.setText(
                    # "Robot: Your last choice was {}. My action was {}. I slipped into a hole. We failed. Please press ENTER to restart.".format(
                    #     action, robot_action))
                else:
                    system_prompt = "You failed. Please press ENTER to restart."
                    robot_prompt = "Your last choice was {}. I followed your choice. I slipped into a hole.".format(
                        human_action_name)
                    textbox.setText(robot=robot_prompt, system=system_prompt)
                    self.window_surface.blit(self.elf_images[0], (256 + map_size, 0))
                    # textbox.setText(
                    #     "Robot: Your last choice was {}. My action was {}. I slipped into a hole. We failed. Please press ENTER to restart.".format(
                    #         action, action))
            elif desc[bot_row][bot_col] == b"G":
                if round_num in CONDITION['practice']:
                    # textbox.setText(
                    #     "Robot: Your last choice was {}. My action was {}. We successfully reached the goal.Please ask the experimenter to start a new game.".format(
                    #         action, action))
                    system_prompt = "You successfully reached the goal.Please ask the experimenter to start a new game."
                    robot_prompt = "Your last choice was {}. I followed your choice.".format(
                        human_action_name)
                    textbox.setText(robot=robot_prompt, system=system_prompt)
                    self.window_surface.blit(self.elf_images[0], (256 + map_size, 0))
                elif robot_type == 2 or robot_type == 4:
                    # if robot_action == None:
                    #     robot_action = robot_action_name
                    # textbox.setText(
                    #     "Robot: Your last choice was {}. My action was {}. We successfully reached the goal. Please finish the survey and ask the experimenter to start a new game.".format(
                    #         action, robot_action))
                    system_prompt = "You successfully reached the goal.Please ask the experimenter to start a new game."
                    if human_action_name == robot_action_name:
                        robot_prompt = "Your last choice was {}. I followed your choice.".format(
                            human_action_name, robot_action_name)
                    else:
                        robot_prompt = "Your last choice was {}. My action was {}.".format(
                            human_action_name, robot_action_name)
                    textbox.setText(robot=robot_prompt, system=system_prompt)
                    self.window_surface.blit(self.elf_images[0], (256 + map_size, 0))
                else:
                    system_prompt = "You successfully reached the goal.Please ask the experimenter to start a new game."
                    robot_prompt = "Your last choice was {}. I followed your choice.".format(
                        human_action_name)
                    textbox.setText(robot=robot_prompt, system=system_prompt)
                    self.window_surface.blit(self.elf_images[0], (256 + map_size, 0))
            else:
                if not timeout:
                    self.window_surface.blit(elf_img, cell_rect)
                # if round_num in CONDITION['practice']:
                #     textbox.setText(robot=
                #     "Your last choice was {}. I followed your choice. n Here I may show you some notifications in the formal study. Please pay attention.".format(
                #         human_action_name))
                #     self.window_surface.blit(self.elf_images[0], (128 + min(64 * self.ncol, 512), 0))
                # else:
                if robot_type:
                    textbox.setText(robot=robot_prompt, system=system_prompt)
                    self.window_surface.blit(self.elf_images[0], (256 + map_size, 0))
                elif human_direction != None:
                    textbox.setText(
                        robot="Your last choice was {}. I followed your choice.".format(human_action_name))
                    self.window_surface.blit(self.elf_images[0], (256 + map_size, 0))
                else:
                    textbox.setText()

        # pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
            round_num = 0
    ):
        super().reset()

        self.render(round_num, None, None, self.world_state)
        return self.world_state

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


class InverseFrozenLakeEnv(FrozenLakeEnvInterface):
    def reward(self, augmented_state, robot_action, human_action=None):
        position, last_position, human_slippery, robot_slippery = augmented_state[:4]
        # Get reward based on the optimality of the human action and the turn number
        # TODO: add penalty if robot takes control etc.
        curr_row = position // self.ncol
        curr_col = position % self.ncol
        last_row = last_position // self.ncol
        last_col = last_position % self.ncol
        reward = 1
        detect = None
        if human_action:
            human_accept, detect, human_choice = human_action
        if detect == 1:
            reward = 2
        if self.desc[curr_row, curr_col] in b'HS' or \
                (self.desc[last_row, last_col] in b'HS' and position == 0 and self.move(last_position,
                                                                                        robot_action[1]) != 0) or\
                (self.desc[last_row, last_col] in b'HS' and robot_action[0] == 0):
            reward = 10
        elif self.desc[curr_row, curr_col] in b'G':
            reward = -30
        return reward


class Driver:
    def __init__(self, env, solver, num_steps, simulated_human, agent=None, max_detection=5):
        """
        Initializes a driver : uses particle filter to maintain belief over hidden states,
        and uses POMCP to determine the optimal robot action

        :param env: (type: Environment) Instance of the Mastermind environment
        :param solver: (type: POMCPSolver) Instance of the POMCP Solver for the robot policy
        :param num_steps: (type: int) number of actions allowed -- I think it's the depth of search in the tree
        :param simulated_human: (type: SimulatedHuman) the simulated human model
        """
        self.env = env
        self.solver = solver
        self.num_steps = num_steps
        self.simulated_human = simulated_human
        self.agent = agent
        self.max_detection = max_detection

    def invigorate_belief(self, current_human_action_node, parent_human_action_node, robot_action, human_action, env):
        """
        Invigorates the belief space when a new human action node is created
        Updates the belief to match the world state, whenever a new human action node is created
        :param current_human_action_node:
        :param parent_human_action_node:
        :param robot_action:
        :param human_action:
        :param env:
        :return:
        """
        # Parent human action node is the h node (root of the current search tree).
        # Current human action node is the hao node.

        for belief_state in parent_human_action_node.belief:
            # Update the belief world state for the current human action node
            # if the belief of the parent human action node is the same as the actual world state

            # Update parent belief state to match world state (i.e., after robot action)
            belief_state = env.augmented_state_transition(belief_state, robot_action, None)
            # if belief_state[:2] == env.world_state[:2]:
            if belief_state[:6] == env.world_state:
                next_augmented_state = env.augmented_state_transition(belief_state, None, human_action)
                current_human_action_node.update_belief(next_augmented_state)
            else:
                print("Node belief is empty!!! Particle Reinvigoration failed!!!")

    def updateBeliefWorldState(self, human_action_node, env):
        """
        Updates the world state in the belief if there are any discrepancies...
        # TODO: Not sure if I need this... In their POMCP code, I don't think they use this ...
        :param human_action_node:
        :param env:
        :return:
        """
        if len(human_action_node.belief) == 0:
            print("Node belief is empty!!!")
            return
        # Update the belief (i.e., all particles) in the current node to match the current world state
        if (human_action_node.belief[0][0] != env.world_state[0]) and (
                human_action_node.belief[0][1] != env.world_state[1]) and \
                (human_action_node.belief[0][2] != env.world_state[2]) and (
                human_action_node.belief[0][3] != env.world_state[3]) and \
                (human_action_node.belief[0][4] != env.world_state[4]) and (
                human_action_node.belief[0][5] != env.world_state[5]):
            human_action_node.belief = [[env.world_state[0], env.world_state[1], env.world_state[2], env.world_state[3],
                                         env.world_state[4], env.world_state[5], belief[6], belief[7]] for belief in
                                        human_action_node.belief]

    def updateBeliefChiH(self, human_action_node, human_action):
        """
        Updates the human capability in belief based on the human's action
        TODO: In their work, they update the chi_h_belief matrix based on whether the human demonstrates a failure as they have access to the decision outcome in each turn.
        I can either only update the capability after the end of each round based on the number of errors they made
        or assume that there is an oracle telling the robot how well the human is doing in each
        action. I need to figure out how to update the robot's belief of human capability based on the human action.
        I might also need to take the state information into consideration
        :param human_action_node:
        :param human_action:
        :return:
        """
        # TODO: I am currently updating the human capability after every turn (assuming that the robot has access
        #  to an oracle that determines the optimality of the user's suggestion after every turn).

        # Here, I use the same update as in the augmented_state_transition function in the env.
        # In the original code, they only update in case of failure here with the actual human action,
        # whereas in the env they use intended human action for the update and update both in the case of success and failure.
        # It makes sense here that they only use the actual human action (which is the observation).

        human_accept, detect, human_choice_idx = human_action  # human accept: 0:no-assist, 1:accept, 2:reject

        for belief in human_action_node.belief:
            if human_accept != 0:  # In case of robot assistance
                # Update trust
                belief[6][human_accept - 1] += 1  # index 0 is acceptance count, index 1 is rejection count

    def updateRootCapabilitiesBelief(self, root_node, current_node):
        """
        Updates the root belief about capabilities to the capabilities of the current node.
        TODO: In the tree search, we keep updating the root based on the observation history (basically we truncate the part
        of the tree, before that... Are we updating the belief of that root node to match the capabilities??
        I'm not too sure what this function is doing yet but I know they're using particle filter to reprsent the belief

        :param root_node:
        :param current_node:
        :return:
        """
        initial_world_state = copy.deepcopy(
            self.env.world_state)  # TODO: Ensure you reset the env. with the correct answer for the next round.
        root_node.belief = []
        num_samples = 10000
        # Sample belief_trust and belief_capability from a distribution
        sampled_beliefs = random.sample(current_node.belief, num_samples) if len(
            current_node.belief) > num_samples else current_node.belief
        root_node.belief.extend([[initial_world_state[0], initial_world_state[1], initial_world_state[2],
                                  initial_world_state[3], initial_world_state[4], initial_world_state[5],
                                  current_node_belief[6], current_node_belief[7]] for current_node_belief in
                                 sampled_beliefs])

    def finalCapabilityCalibrationScores(self, human_action_node):
        """
        Returns the average capability calibration scores from particles sampled from the input human action node
          TODO: Not sure if we need this... --> Not using this for now
          :param human_action_node: the human action node from which particles are sampled to be evaluated
          :return: expected robot capability calibration score, human capability calibration score
        """
        num_samples = 10000
        sampled_beliefs = random.sample(human_action_node.belief, num_samples) if len(
            human_action_node.belief) > num_samples else human_action_node.belief

        total_robot_capability_score = 0
        total_human_capability_score = 0
        for belief in sampled_beliefs:
            total_robot_capability_score += self.env.robotCapabilityCalibrationScore(
                belief)  # TODO: Need to implement this in env --> Not using this for now
            total_human_capability_score += self.env.humanCapabilityCalibrationScore(
                belief)  # TODO: Need to implement this in env --> Not using this for now

        return total_robot_capability_score / len(sampled_beliefs), total_human_capability_score / len(sampled_beliefs)

    def beliefRewardScore(self, belief):
        """
        Returns the reward belief score for the current belief
        :param belief:
        :return:
        """
        raise NotImplementedError

    def render_score(self, tmp, round_num, final_env_reward, step, detecting_num):
        if round_num == 0:
            x = font.render(
                "Demo {}".format(round_num + 1),
                True, (0, 0, 0))
            tmp.blit(x, (5, 5))
            x = font.render(
                "Score: {}".format(int(final_env_reward)),
                True, (0, 0, 0))
            tmp.blit(x, (5, 30))
            x = font.render(
                "Steps Left: {}".format(max(self.num_steps - step, 0)),
                True, (0, 0, 0))
            tmp.blit(x, (5, 55))
            x = font.render(
                "ID: " + username,
                True, (0, 0, 0))
            tmp.blit(x, (5, 80))
            x = font.render(
                "Detections Left: {}".format(self.max_detection - detecting_num),
                True, (0, 0, 0))
            tmp.blit(x, (5, 105))
        elif round_num in CONDITION['practice']:
            x = font.render(
                "Practice {}".format(round_num),
                True, (0, 0, 0))
            tmp.blit(x, (5, 5))
            x = font.render(
                "Score: {}".format(int(final_env_reward)),
                True, (0, 0, 0))
            tmp.blit(x, (5, 30))
            x = font.render(
                "Steps Left: {}".format(max(self.num_steps - step, 0)),
                True, (0, 0, 0))
            tmp.blit(x, (5, 55))
            x = font.render(
                "ID: " + username,
                True, (0, 0, 0))
            tmp.blit(x, (5, 80))
            x = font.render(
                "Detections Left: {}".format(self.max_detection - detecting_num),
                True, (0, 0, 0))
            tmp.blit(x, (5, 105))
        else:
            x = font.render(
                "Round {}".format(round_num - 3),
                True, (0, 0, 0))
            tmp.blit(x, (5, 5))
            x = font.render(
                "Score: {}".format(final_env_reward),
                True, (0, 0, 0))
            tmp.blit(x, (5, 30))
            x = font.render(
                "Steps Left: {}".format(max(self.num_steps - step, 0)),
                True, (0, 0, 0))
            tmp.blit(x, (5, 55))
            x = font.render(
                "ID: " + username,
                True, (0, 0, 0))
            tmp.blit(x, (5, 80))
            x = font.render(
                "Detections Left: {}".format(self.max_detection - detecting_num),
                True, (0, 0, 0))
            tmp.blit(x, (5, 105))
        pygame.display.update()

    def execute(self, round_num, debug_tree=False):
        """
        Executes one round of search with the POMCP policy
        :param round_num: (type: int) the round number of the current execution
        :return: (type: float) final reward from the environment
        """
        robot_actions = []
        human_actions = []
        all_states = []

        # create a deep copy of the env and the solver
        env = self.env
        solver = self.solver

        print("Execute round {} of search".format(round_num))
        start_time = time.time()
        final_env_reward = 0
        step = 0

        # Initial human action
        robot_action = (0, None)  # No interruption
        init_human_action = self.simulated_human.simulateHumanAction(env.world_state, robot_action)
        # init_human_action = tuple([0] + [int(i) for i in input("Enter human action separated by comma: ").split(',')])

        # Keyboard input
        action = None
        detecting = 0
        is_accept = 0
        truncated = False
        detection_num = 0

        tmp = env.window_surface
        pygame.font.init()
        font = pygame.font.Font(pygame.font.match_font('calibri'), 17)

        history = []
        data = {}

        while action == None:

            # Show scores
            tmp.fill((255, 255, 255), rect=(0, 0, 256, 256))
            self.render_score(tmp, round_num, final_env_reward, step, detection_num)

            for event in pygame.event.get():
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_BACKSPACE:
                        if detecting:
                            detecting = 0
                            env.render(round_num, None, None, env.world_state)
                        else:
                            detecting = 1
                            env.render(round_num, None, None, env.world_state)
                    if event.key == pygame.K_LEFT:
                        action = 0
                    if event.key == pygame.K_RIGHT:
                        action = 2
                    if event.key == pygame.K_UP:
                        action = 3
                    if event.key == pygame.K_DOWN:
                        action = 1
        is_accept = 0
        human_action = tuple([is_accept, detecting, action])
        if detecting:
            detection_num += 1

        data['human_action'] = human_action

        # print("Human Initial Action: ", init_human_action)
        last_human_action = human_action
        last_robot_action = [0, None]
        # Here we are adding to the tree as this will become the root for the search in the next turn
        human_action_node = HumanActionNode(env)
        # This is where we call invigorate belief... When we add a new human action node to the tree
        self.invigorate_belief(human_action_node, solver.root_action_node, robot_action, human_action, env)
        solver.root_action_node = human_action_node
        env.world_state = env.world_state_transition(env.world_state, robot_action, human_action)
        # Extra -1 point, keeping for pilot TODO: Remove it in user study
        if round_num in CONDITION['pomcp_inverse']:
            final_env_reward -= env.reward(env.world_state, robot_action, human_action)
        else:
            final_env_reward += env.reward(env.world_state, robot_action, human_action)
        all_states.append(env.world_state[0])
        human_actions.append(human_action)

        while True:
            # One extra step penalty if using detection
            if last_human_action[1] == 1:
                step += 2
            else:
                step += 1
            # if step == 19:
            #     print()
            t = time.time()
            # Show scores
            tmp.fill((255, 255, 255), rect=(0, 0, 256, 256))
            self.render_score(tmp, round_num, final_env_reward, step, detection_num)

            if round_num == 0:
                robot_action_type = 0
                if env.world_state[0] == 9:
                    robot_action_type = 2
                if env.world_state[0] == 3:
                    robot_action_type = 1
            elif round_num == 1:
                # print("practice")
                if last_human_action[1] == 0:
                    _ = solver.search()
                robot_action_type = 0
            elif round_num in [2, 3]:
                # epsilon-greedy pomcp agent
                if last_human_action[1] == 1:
                    robot_action_type = 0
                else:
                    robot_action_type = solver.search()
                    if np.random.uniform() < 0.5:
                        robot_action_type = np.random.choice([0, 1, 2, 3, 4],
                                                             p=[0.5, 0.5 / 4, 0.5 / 4, 0.5 / 4, 0.5 / 4])
                        # print("Random", robot_action_type)
            elif round_num in CONDITION['pomcp'] + CONDITION['pomcp_inverse']:
                # print("pomcp")
                if last_human_action[1] == 1:
                    robot_action_type = 0
                else:
                    robot_action_type = solver.search()  # One iteration of the POMCP search  # Here the robot action indicates the type of assistance
            else:
                _ = solver.search()  # Make the heuristic agent execution slower
                if last_robot_action[0] or human_action[1]:
                    robot_action_type = 0  # Cannot interrupt twice successively
                else:
                    robot_action_type = agent.get_action(env)

            robot_action = env.get_robot_action(env.world_state[:6], robot_action_type)
            robot_action_node = solver.root_action_node.robot_node_children[robot_action[0]]

            if debug_tree:
                visualize_tree(solver.root_action_node)

            if robot_action_node == "empty":
                # We're not adding to the tree though here
                # It doesn't matter because we are going to update the root from h to hao
                robot_action_node = RobotActionNode(env)

            if round_num in CONDITION['practice']:
                data['condition'] = 'practice'
            elif round_num in CONDITION['pomcp']:
                data['condition'] = 'pomcp'
            elif round_num in CONDITION['pomcp_inverse']:
                data['condition'] = 'pomcp_inverse'
            elif round_num in CONDITION['interrupt']:
                data['condition'] = 'interrupt'
            elif round_num in CONDITION['take_control']:
                data['condition'] = 'take_control'

            # print("Robot Action: ", robot_action)
            last_robot_action = robot_action
            # data['robot_action'] = [int(robot_action[0]), robot_action[1]]
            data['robot_action'] = robot_action
            # if last_human_action[1] == 0 and robot_action[1] == last_human_action[2]:
            #     robot_action = (0, None)

            # Update the environment
            env.world_state = env.world_state_transition(env.world_state, robot_action, None)
            data['last_state'] = env.world_state[1]
            data['current_state'] = env.world_state[0]
            if env.desc[env.world_state[0] // env.ncol, env.world_state[0] % env.ncol] in b'S':
                data['type'] = 'slippery'
            elif env.desc[env.world_state[0] // env.ncol, env.world_state[0] % env.ncol] in b'H':
                data['type'] = 'hole'
            elif env.world_state == 63:
                data['type'] = 'goal'
            else:
                data['type'] = 'ice'
            robot_action_node.position = env.world_state[0]

            all_states.append(env.world_state[0])

            # print("World state after robot action: ", env.world_state)
            curr_row = env.world_state[0] // env.ncol
            curr_col = env.world_state[0] % env.ncol
            last_row = env.world_state[1] // env.ncol
            last_col = env.world_state[1] % env.ncol
            if env.desc[curr_row, curr_col] in b'HS' or (
                    env.world_state[0] == 0 and robot_action[0] in [2, 4] and env.move(env.world_state[1], robot_action[1]) != 0) or (
                    env.desc[last_row, last_col] in b'HS' and robot_action[0] == 0):
                truncated = True
                # env.render(round_num, None, None, env.world_state, truncated=truncated)

            # if last_human_action[1] == 0 and robot_action[1] == last_human_action[2]:
            #     robot_action = (0, None)
            env.render(round_num, last_human_action, robot_action, env.world_state, truncated=truncated)
            # for event in pygame.event.get():
            #     if event.type == pygame.KEYUP:
            #         break
            # print("Robot map")
            # env.render(env.desc)
            # print("Time: ", time.time() - t)

            if round_num in CONDITION['pomcp_inverse']:
                final_env_reward -= env.reward(env.world_state, robot_action, human_action)
            else:
                final_env_reward += env.reward(env.world_state, robot_action, human_action)

            data['score'] = final_env_reward
            history.append(copy.deepcopy(data))
            data = {}
            # print("Reward:", env.reward(env.world_state, robot_action, human_action))

            # Terminates if goal is reached
            if env.isTerminal(env.world_state):
                print("Final reward: ", final_env_reward)
                # Show scores
                tmp.fill((255, 255, 255), rect=(0, 0, 256, 256))
                self.render_score(tmp, round_num, final_env_reward, step, detection_num)
                # Wait to open next game
                wait = True
                while wait:
                    for event in pygame.event.get():
                        # Show scores
                        # tmp.fill((255, 255, 255), rect=(0, 0, 128, 256))
                        # self.render_score(tmp, round_num, final_env_reward, step)
                        if event.type == pygame.KEYUP:
                            if event.key == pygame.K_SPACE:
                                wait = False
                    # env.render(round_num, None, None, env.world_state)
                    # tmp.fill((255, 255, 255), rect=(0, 0, 128, 256))
                    # self.render_score(tmp, round_num, final_env_reward, step)
                break

            # Terminates if reaching the maximum step
            if step >= self.num_steps:
                env.render(round_num, None, None, env.world_state, timeout=True)
                # Show scores
                tmp.fill((255, 255, 255), rect=(0, 0, 256, 256))
                self.render_score(tmp, round_num, final_env_reward, step, detection_num)
                wait = True
                while wait:
                    # Show scores
                    tmp.fill((255, 255, 255), rect=(0, 0, 256, 256))
                    self.render_score(tmp, round_num, final_env_reward, step, detection_num)
                    for event in pygame.event.get():

                        # Show scores
                        tmp.fill((255, 255, 255), rect=(0, 0, 256, 256))
                        self.render_score(tmp, round_num, final_env_reward, step, detection_num)
                        if event.type == pygame.KEYUP:
                            if event.key == pygame.K_SPACE:
                                wait = False

                break

            # We finally use the real observation / human action (i.e., from the simulated human model)

            # Note here that it is not the augmented state
            # (the latent parameters are already defined in the SimulatedHuman model I think)
            human_action = self.simulated_human.simulateHumanAction(env.world_state, robot_action)
            if robot_action[0] or truncated:
                pause = True
            else:
                pause = False
            action = None
            pygame.event.clear() #Clear the pygame.event caches so it won't move multiple steps if pressing keys multiple times
            is_accept = 0
            while action == None:

                # Show scores
                tmp.fill((255, 255, 255), rect=(0, 0, 256, 256))
                self.render_score(tmp, round_num, final_env_reward, step, detection_num)

                for event in pygame.event.get():
                    if event.type == pygame.KEYUP:
                        if event.key == pygame.K_RETURN and pause:
                            pause = False
                            if truncated:
                                truncated = False
                            env.render(round_num, None, None, env.world_state,
                                       truncated=truncated)  # round_num, human_action, robot_action, world_state, end_detecting=False, truncated=False, timeout=False
                        if event.key == pygame.K_BACKSPACE and not truncated:
                            if detecting:
                                detecting = 0
                                pause = False
                                env.render(round_num, None, None, env.world_state, end_detecting=1)
                            elif detection_num < self.max_detection:
                                detecting = 1
                                pause = False
                                env.render(round_num, (None, 1, None), None, env.world_state)
                            else:
                                detecting = 1
                                env.render(round_num, None, None, env.world_state, end_detecting=2)
                            # env.render(None, None, detecting, None, False)
                        if not detecting or (detecting and detection_num < self.max_detection):
                            if event.key == pygame.K_LEFT and not pause:
                                action = 0
                            if event.key == pygame.K_RIGHT and not pause:
                                action = 2
                            if event.key == pygame.K_UP and not pause:
                                action = 3
                            if event.key == pygame.K_DOWN and not pause:
                                action = 1
                        elif detecting and detection_num >= self.max_detection:
                            if event.key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
                                pause = True
                                env.render(round_num, None, None, env.world_state, end_detecting=2)
                # if action:
                #     # human_action = [detecting, action]
                #     # print(detecting, action)
                #     break
            if detecting and detection_num < self.max_detection:
                detection_num += 1

            # human_action = [int(i) for i in input("Enter human action separated by comma: ").split(',')]
            if (robot_action[0] == 1 or robot_action[0] == 3) and detecting != 1 and action == \
                    last_human_action[2]:
                is_accept = 2
            elif (robot_action[0] == 2 or robot_action[0] == 4) and detecting != 1 and abs(
                    last_human_action[2] - robot_action[1]) == 2:
                is_accept = 2
            elif robot_action[0] == 0 or detecting == 1:
                is_accept = 0
            else:
                is_accept = 1
            human_action = tuple([is_accept, detecting, action])

            # ans = input("Change human action {}? Y/N: ".format(human_action))
            # if ans == "Y":
            #     human_action = tuple([int(i) for i in input("Enter human action separated by comma: ").split(',')])
            # print("Human Action: ", human_action)
            human_action_node = robot_action_node.human_node_children[human_action[1] * 4 + human_action[2]]

            last_human_action = human_action
            data['human_action'] = human_action

            if human_action_node == "empty":
                # Here we are adding to the tree as this will become the root for the search in the next turn
                human_action_node = robot_action_node.human_node_children[
                    human_action[1] * 4 + human_action[2]] = HumanActionNode(env)
                # This is where we call invigorate belief... When we add a new human action node to the tree
                self.invigorate_belief(human_action_node, solver.root_action_node, robot_action, human_action, env)

            # Update the environment
            solver.root_action_node = human_action_node  # Update the root node from h to hao
            env.world_state = env.world_state_transition(env.world_state, robot_action, human_action)
            all_states.append(env.world_state[0])
            # Updates the world state in the belief to match the actual world state
            # The original POMCP implementation in this codebase does not do this...
            # Technically if all the belief updates are performed correctly, then there's no need for this.
            self.updateBeliefWorldState(human_action_node, env)

            # Updates robot's belief of the human capability based on human action
            # TODO: We cannot really evaluate the outcome at every turn... but I'm still updating based on choice optimality
            #  So should I only update human capability after the round is over?
            #  Should this come before root node transfer? It dm in their case
            self.updateBeliefChiH(human_action_node, human_action)  # For now I'm updating every turn.
            # print("Human action: ", human_action)

            # print("World state after human action: ", env.world_state)

            # print("Human map")
            # env.render(env.desc)

            # Prints belief over hidden state theta (debugging)
            # temp_belief = [0] * len(env.reward_space)  # TODO: Need to implement env.reward_space
            # for particle in solver.root_action_node.belief:
            #     temp_belief[particle[1]] += 1
            # print("Belief at selected human action node: ", temp_belief)
            # print("Number of particles at selected human action node: ", len(solver.root_action_node.belief))  # TODO
            # # print('belief reward score for true theta: ', self.beliefRewardScore(solver.root_action_node.belief))  # TODO

            robot_actions.append(robot_action)
            human_actions.append(human_action)

            # print("Root Node Value: ", solver.root_action_node.value)
            # print("===================================================================================================")

            # # Terminates if goal is reached
            # if env.isTerminal(env.world_state):
            #     break

            # Transfer current capabilities beliefs to the next round
        self.updateRootCapabilitiesBelief(self.solver.root_action_node, solver.root_action_node)

        print("===================================================================================================")
        print("Round {} completed!".format(round_num))
        print("Time taken:")
        print("{} seconds".format(time.time() - start_time))
        print('Robot Actions: {}'.format(robot_actions))
        print('Human Actions: {}'.format(human_actions))
        # print("final world state for the round: ")

        # TODO: Fix this and calculate from env?
        # final_env_reward = env.final_reward([env.true_world_state, env.human_trust, env.human_capability,
        #                                      env.human_aggressiveness])

        return final_env_reward, history

# function to write json files
def write_json(path, data, indent=4):
    class npEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.int32):
                return int(obj)
            return json.JSONEncoder.default(self, obj)
    with open(path, 'w') as file:
        json.dump(data, file, indent=indent, cls=npEncoder)

if __name__ == '__main__':

    pygame.init()
    pygame.display.init()
    pygame.display.set_caption("Frozen Lake")
    window_surface = pygame.display.set_mode((min(100 * 8, 1024) + 256 * 2, min(100 * 8, 800) + 300))
    pygame.font.init()
    font = pygame.font.Font(None, 30)
    x = font.render(
        "Participant ID: {}. ".format(username), True,
        (0, 0, 0))
    window_surface.fill((255, 255, 255))
    window_surface.blit(x, (10, 20))
    x = font.render(
        "Please finish the pre-experiment questionnaire and ask the experimenter".format(username), True,
        (0, 0, 0))
    window_surface.blit(x, (10, 50))
    x = font.render(
        "to start the study.".format(username), True,
        (0, 0, 0))
    window_surface.blit(x, (10, 80))
    pygame.display.flip()
    print("Participant ID", username)
    input("Press Enter to continue...")

    # Choose heuristic agent type
    # exp_type = random.choice([0, 2])  # Without explanation or with
    exp_type = 2 #With explanation
    explantion = exp_type == 2
    print(explantion)

    # Set appropriate seeds
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    # (set different values for each test)
    true_trust = [(5, 50), (10, 40), (18, 40), (24, 36), (35, 35), (40, 65), (45, 20), (45, 56),
                  (40, 45), (99, 1)]
    # true_trust = [(5, 50), (35, 35), (99, 1)]
    true_capability = 0.85  # fixed - parameter (assume known??) at the start of the study

    # The following two parameters are for human behavior. They are currently not used.
    human_behavior = "rational"  # TODO: they use this in the observation function in the environment
    beta = 0.9  # Boltzmann rationality parameter (for human behavior)

    # factors for POMCP
    gamma = 0.99  # gamma for terminating rollout based on depth in MCTS
    c = 20  # 400  # exploration constant for UCT (taken as R_high - R_low)
    e = 0.1  # For epsilon-greedy policy

    epsilon = math.pow(gamma, 40)  # tolerance factor to terminate rollout
    num_iter = 100

    num_steps = 80

    # Executes num_tests of experiments

    num_test = 10
    mean_rewards = []
    std_rewards = []
    all_rewards = []
    for n in range(4, num_test):
        start_t = time.time()
        initial_belief = []
        print("*********************************************************************")
        print("Executing test number {}......".format(n))
        print("*********************************************************************")

        # Robot's belief of human parameters
        all_initial_belief_trust = []
        for _ in range(1000):
            all_initial_belief_trust.append((1, 1))

        # Setup Driver

        map_num = mapOrder[n]
        map = MAPS["MAP" + str(map_num)]
        foggy = FOG["MAP" + str(map_num)]
        human_err = HUMAN_ERR["MAP" + str(map_num)]
        robot_err = ROBOT_ERR["MAP" + str(map_num)]
        # robot_err = []
        if n in CONDITION['pomcp_inverse']:
            env = InverseFrozenLakeEnv(desc=map, foggy=foggy, human_err=human_err, robot_err=robot_err,
                                       is_slippery=False, render_mode="human", true_human_trust=true_trust[n],
                                       true_human_capability=true_capability,
                                       true_robot_capability=0.85, beta=beta, c=c, gamma=gamma, seed=SEED,
                                       human_type="epsilon_greedy", round=n)
        else:
            env = FrozenLakeEnvInterface(desc=map, foggy=foggy, human_err=human_err, robot_err=robot_err,
                                         is_slippery=False, render_mode="human", true_human_trust=true_trust[n],
                                         true_human_capability=true_capability,
                                         true_robot_capability=0.85, beta=beta, c=c, gamma=gamma, seed=SEED,
                                         human_type="epsilon_greedy", round=n)

        # Reset the environment to initialize everything correctly
        env.reset(round_num=n)
        init_world_state = env.world_state

        # TODO: Initialize belief: Currently only using the 4 combinations
        for i in range(len(all_initial_belief_trust)):
            initial_belief.append(init_world_state + [list(all_initial_belief_trust[i])] + [true_capability])

        root_node = RootNode(env, initial_belief)
        solver = POMCPSolver(epsilon, env, root_node, num_iter, c)
        simulated_human = SimulatedHuman(env, true_trust=true_trust[n],
                                         true_capability=true_capability,
                                         type="epsilon_greedy",
                                         seed=SEED)

        if n in CONDITION['interrupt']:
            agent = HeuristicAgent(type=exp_type+1)
            driver = Driver(env, solver, num_steps, simulated_human, agent=agent)
        elif n in CONDITION['take_control']:
            agent = HeuristicAgent(type=exp_type+2)
            driver = Driver(env, solver, num_steps, simulated_human, agent=agent)
        else:
            driver = Driver(env, solver, num_steps, simulated_human)
            explantion = None

        # Executes num_rounds of search (calibration)
        num_rounds = 10
        total_env_reward = 0

        # rewards = []
        # for i in range(num_rounds):
        # We should only change the true state of the tiger for every round (or after every termination)
        driver.env.reset(round_num=n)  # Note tiger_idx can be either 0 or 1 indicating left or right door
        env_reward, history = driver.execute(n, debug_tree=False)
        # rewards.append(env_reward)
        total_env_reward += env_reward

        userdata[str(n)] = {'history': history,
                            'duration': time.time() - start_t,
                            'explanation': explantion,
                            'optimal_path': len(env.find_shortest_path(env.desc, env.hole + env.slippery, 0, env.ncol))}

        print("===================================================================================================")
        print("===================================================================================================")
        print("Average environmental reward after {} rounds:{}".format(num_rounds,
                                                                       total_env_reward / float(num_rounds)))
        print("Num Particles: ", len(driver.solver.root_action_node.belief))
        all_rewards.append(env_reward)
        # mean_rewards.append(np.mean(rewards))
        # std_rewards.append(np.std(rewards))

        write_json(filename, userdata)

    print("===================================================================================================")
    print("===================================================================================================")
    print(mean_rewards, std_rewards)
    print("===================================================================================================")
    print("===================================================================================================")

    all_rewards = np.array(all_rewards)

        #
        # with open("files/pomcp_{}_{}_map_{}/{}.npy".format(num_iter, 40, map_num, SEED), 'wb') as f:
        #     np.save(f, all_rewards)