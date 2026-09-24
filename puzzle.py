from __future__ import division
from __future__ import print_function

import sys
import math
import time
import queue as Q
import resource


#### SKELETON CODE ####
## The Class that Represents the Puzzle
class PuzzleState(object):
    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n*n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n*n)):
            raise Exception("Config contains invalid/duplicate entries : ", config)

        self.n        = n
        self.cost     = cost
        self.parent   = parent
        self.action   = action
        self.config   = config
        self.children = []

        # Get the index and (row, col) of empty block
        self.blank_index = self.config.index(0)

    def display(self):
        """ Display this Puzzle state as a n*n board """
        for i in range(self.n):
            print(self.config[self.n*i : self.n*(i+1)])

    def _move(self, valid, offset, action):
        if not valid:
            return None
        new_config = list(self.config)
        target = self.blank_index + offset
        new_config[self.blank_index], new_config[target] = new_config[target], new_config[self.blank_index]
        return PuzzleState(new_config, self.n, parent=self, action=action, cost=self.cost + 1)

    def move_up(self):
        """ 
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """
        return self._move(self.blank_index >= self.n, -self.n, "Up")
      
    def move_down(self):
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        return self._move(self.blank_index < self.n * (self.n - 1), self.n, "Down")
      
    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        return self._move(self.blank_index % self.n != 0, -1, "Left")

    def move_right(self):
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        return self._move(self.blank_index % self.n != self.n - 1, 1, "Right")
      
    def expand(self):
        """ Generate the child nodes of this node """
        
        # Node has already been expanded
        if len(self.children) != 0:
            return self.children
        
        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        return self.children

# Function that Writes to output.txt

### Students need to change the method to have the corresponding parameters
def writeOutput(state, nodes_expanded, max_search_depth, start_time, ram_start):
    ### Student Code Goes here
    path = []
    node = state
    while node.parent is not None:
        path.append(node.action)
        node = node.parent
    path.reverse()

    running_time = time.time() - start_time
    max_ram_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - ram_start) / (2 ** 20)

    with open("output.txt", "w") as f:
        f.write(
            "path_to_goal: %s\n"
            "cost_of_path: %d\n"
            "nodes_expanded: %d\n"
            "search_depth: %d\n"
            "max_search_depth: %d\n"
            "running_time: %.8f\n"
            "max_ram_usage: %.8f\n"
            % (path, len(path), nodes_expanded, state.cost, max_search_depth, running_time, max_ram_usage)
        )

def bfs_search(initial_state):
    """BFS search"""
    ### STUDENT CODE GOES HERE ###
    start_time = time.time()
    ram_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    frontier = Q.Queue()
    frontier.put(initial_state)
    frontier_config = {tuple(initial_state.config)}

    explored = set()
    nodes_expanded = 0
    max_search_depth = 0

    while not frontier.empty():
        state = frontier.get()
        config = tuple(state.config)
        frontier_config.discard(config)
        explored.add(config)

        if test_goal(state):
            writeOutput(state, nodes_expanded, max_search_depth, start_time, ram_start)
            return True

        nodes_expanded += 1
        for neighbor in state.expand():
            neighbor_config = tuple(neighbor.config)
            if neighbor_config not in explored and neighbor_config not in frontier_config:
                frontier.put(neighbor)
                frontier_config.add(neighbor_config)
                if neighbor.cost > max_search_depth:
                    max_search_depth = neighbor.cost

    return False

def dfs_search(initial_state):
    """DFS search"""
    ### STUDENT CODE GOES HERE ###
    start_time = time.time()
    ram_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    frontier = [initial_state]
    frontier_config = {tuple(initial_state.config)}

    explored = set()
    nodes_expanded = 0
    max_search_depth = 0

    while frontier:
        state = frontier.pop()
        config = tuple(state.config)
        frontier_config.discard(config)
        explored.add(config)

        if test_goal(state):
            writeOutput(state, nodes_expanded, max_search_depth, start_time, ram_start)
            return True

        nodes_expanded += 1

        for neighbor in reversed(state.expand()):
            neighbor_config = tuple(neighbor.config)
            if neighbor_config not in explored and neighbor_config not in frontier_config:
                frontier.append(neighbor)
                frontier_config.add(neighbor_config)
                if neighbor.cost > max_search_depth:
                    max_search_depth = neighbor.cost

    return False

def A_star_search(initial_state):
    """A * search"""
    ### STUDENT CODE GOES HERE ###
    start_time = time.time()
    ram_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    frontier = Q.PriorityQueue()
    counter = 0
    initial_cost = calculate_total_cost(initial_state)
    frontier.put((initial_cost, counter, initial_state))
    frontier_cost = {tuple(initial_state.config): initial_cost}

    explored = set()
    nodes_expanded = 0
    max_search_depth = 0

    while not frontier.empty():
        _, _, state = frontier.get()
        config = tuple(state.config)

        # Skip stale duplicate entries left behind by decrease-key updates
        if config in explored:
            continue

        explored.add(config)
        frontier_cost.pop(config, None)

        if test_goal(state):
            writeOutput(state, nodes_expanded, max_search_depth, start_time, ram_start)
            return True

        nodes_expanded += 1
        for neighbor in state.expand():
            neighbor_config = tuple(neighbor.config)
            if neighbor_config in explored:
                continue

            neighbor_cost = calculate_total_cost(neighbor)
            if neighbor_config not in frontier_cost or neighbor_cost < frontier_cost[neighbor_config]:
                counter += 1
                frontier.put((neighbor_cost, counter, neighbor))
                frontier_cost[neighbor_config] = neighbor_cost
                if neighbor.cost > max_search_depth:
                    max_search_depth = neighbor.cost

    return False

def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    ### STUDENT CODE GOES HERE ###
    return state.cost + sum(
        calculate_manhattan_dist(idx, value, state.n)
        for idx, value in enumerate(state.config)
    )

def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile"""
    ### STUDENT CODE GOES HERE ###
    if value == 0:
        return 0
    target_row, target_col = divmod(value, n)
    current_row, current_col = divmod(idx, n)
    return abs(target_row - current_row) + abs(target_col - current_col)

def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    ### STUDENT CODE GOES HERE ###
    return puzzle_state.config == list(range(puzzle_state.n * puzzle_state.n))

# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    search_mode = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = list(map(int, begin_state))
    board_size  = int(math.sqrt(len(begin_state)))
    hard_state  = PuzzleState(begin_state, board_size)
    start_time  = time.time()
    
    if   search_mode == "bfs": bfs_search(hard_state)
    elif search_mode == "dfs": dfs_search(hard_state)
    elif search_mode == "ast": A_star_search(hard_state)
    else: 
        print("Enter valid command arguments !")
        
    end_time = time.time()
    print("Program completed in %.3f second(s)"%(end_time-start_time))

if __name__ == '__main__':
    main()
