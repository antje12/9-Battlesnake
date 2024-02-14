# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# or in the "license" file accompanying this file. This file is distributed 
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
# express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import math
import random

import numpy as np
from .utils import get_random_coordinates, generate_coordinate_list_from_binary_map

class Food:
    '''
    Parameters
    ----------
    map_size: (int, int)
    food_spawn_location: [(int, int)] optional
        Parameter to force food to spawn in certain positions. Used for testing
        Food will spawn in the coordinates provided in the list until the list is exhausted.
        After the list is exhausted, food will be randomly spawned
    '''
    FOOD_SPAWN_CHANCE = 0.15
    def __init__(self, map_size, food_spawn_locations=[]):
        self.map_size = map_size
        self.locations_map = np.zeros(shape=(map_size[0], map_size[1]))

        self.food_spawn_locations = food_spawn_locations

    @classmethod
    def make_from_list(cls, map_size, food_list):
        '''
        Class function to build the Food class.
        Parameters
        ---------
        map_size: (int, int)
        food_list: [(int, int)]
            Coordinates of the food locations
        '''
        cls = Food(map_size)
        for food in food_list:
            i, j = food
            cls.locations_map[i, j] = 1
        return cls

    def spawn_food(self, snake_map):
        '''
        Helper function to generate another food.
        
        Parameters:
        ----------
        snake_map, np.array(map_size[0], map_size[1], 1)
            The map of the location of each snake, generated by Snakes.get_snake_binary_map
        '''
        if len(self.food_spawn_locations) > 0:
            locations = [self.food_spawn_locations[0]]
            self.food_spawn_locations = self.food_spawn_locations[1:]
        else:
            snake_locations = generate_coordinate_list_from_binary_map(snake_map)
            locations = get_random_coordinates(self.map_size, 1, excluding=snake_locations)
        for location in locations:
            self.locations_map[location[0], location[1]] = 1
        
    def end_of_turn(self, snake_locations):
        '''
        Function to be called at the end of each step. 
        Adapted from 
        https://github.com/BattlesnakeOfficial/rules/blob/44b6b946661d42401f5a33b74303cd9071d0db18/standard.go#L392
        '''
        if random.random() < self.FOOD_SPAWN_CHANCE:
            self.spawn_food(snake_locations)
                    
    def get_food_map(self):
        '''
        Function to get a binary image of all the present food
        Returns:
        --------
        map np.array
            binary image of size self.map_size indicating the positions
            of the food on the map.
        '''

        return self.locations_map

    def does_coord_have_food(self, coord):
        '''
        Function to check if a coordinate has food.

        Parameters
        ----------
        coord: (int, int)
            Input coordinate to check if food is available in this coordinate
        '''

        return self.locations_map[coord[0], coord[1]] == 1

    def remove_food_from_coord(self, coord):
        '''
        Function to remove a food present at coord
        '''
        self.locations_map[coord[0], coord[1]] = 0
