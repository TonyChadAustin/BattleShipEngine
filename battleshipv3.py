
from itertools import product
from collections import defaultdict
import pickle
import os


class BattleshipEngine:
    def __init__(self, board_size, ships):
        """
        Initialize the battleship engine.
        
        Args:
            board_size: int, size of the square board (e.g., 10 for 10x10)
            ships: list of tuples (ship_name, ship_length)
                   e.g., [("Carrier", 5), ("Battleship", 4), ("Destroyer", 3)]
        """
        self.board_size = board_size
        self.ships = ships
        self.all_states = []
        self.valid_states = []
        
        # Game state tracking
        self.misses = set()  # Set of (x, y) coordinates that are misses
        self.hits = set()    # Set of (x, y) coordinates that are hits
        self.sunk_ships = []  # List of (ship_idx, set of hit coords) for sunk ships
        
        # Undo history
        self.history = []  # List of (action_type, data) tuples
        
        print(f"Initializing {board_size}x{board_size} board with ships: {ships}")
    
    def get_cache_filename(self):
        """
        Generate a descriptive cache filename based on board configuration.
        Uses ship names to uniquely identify custom configurations.
        
        Returns:
            str: filename like "battleship_5x5_Destroyer_Submarine.pkl"
        """
        # Use ship names for more descriptive filenames
        ship_names = "_".join([name for name, data in self.ships])
        # Sanitize filename (remove spaces, special chars)
        ship_names = ship_names.replace(" ", "").replace(",", "")
        filename = f"battleship_{self.board_size}x{self.board_size}_{ship_names}.pkl"
        return filename
        
    def generate_placements(self, ship_length):
        """
        Generate all possible placements for a single linear ship.
        
        Returns:
            list of tuples (x, y, horizontal)
            where horizontal=True means ship extends right, False means down
        """
        placements = []
        
        # Horizontal placements
        for y in range(self.board_size):
            for x in range(self.board_size - ship_length + 1):
                placements.append((x, y, True))
        
        # Vertical placements
        for y in range(self.board_size - ship_length + 1):
            for x in range(self.board_size):
                placements.append((x, y, False))
        
        return placements
    
    def generate_custom_placements(self, cell_offsets):
        """
        Generate all possible placements for a custom-shaped ship.
        
        Args:
            cell_offsets: list of (dx, dy) offsets from origin, e.g., [(0,0), (1,0), (0,1)]
            
        Returns:
            list of tuples (x, y, rotation_idx) where rotation_idx is 0-3
        """
        placements = []
        
        # Generate 4 rotations of the shape
        rotations = [cell_offsets]
        for _ in range(3):
            # Rotate 90 degrees clockwise: (x, y) -> (y, -x)
            last_rotation = rotations[-1]
            rotated = [(dy, -dx) for dx, dy in last_rotation]
            # Normalize to have min coords at (0, 0)
            min_x = min(x for x, y in rotated)
            min_y = min(y for x, y in rotated)
            normalized = [(x - min_x, y - min_y) for x, y in rotated]
            rotations.append(normalized)
        
        # For each rotation, find all valid positions on the board
        for rotation_idx, rotation in enumerate(rotations):
            max_dx = max(x for x, y in rotation)
            max_dy = max(y for x, y in rotation)
            
            for board_y in range(self.board_size - max_dy):
                for board_x in range(self.board_size - max_dx):
                    # Check if all cells fit on board
                    valid = True
                    for dx, dy in rotation:
                        if board_x + dx >= self.board_size or board_y + dy >= self.board_size:
                            valid = False
                            break
                    
                    if valid:
                        placements.append((board_x, board_y, rotation_idx))
        
        return placements
    
    def placement_to_cells(self, x, y, rotation_or_horizontal, ship_data):
        """
        Convert a placement to the set of cells it occupies.
        
        Args:
            x, y: origin position
            rotation_or_horizontal: for linear ships, bool (True=horizontal, False=vertical)
                                   for custom ships, int (rotation index 0-3)
            ship_data: either int (length) or list of (dx, dy) offsets
        
        Returns:
            set of (x, y) tuples
        """
        cells = set()
        
        if isinstance(ship_data, int):
            # Linear ship
            ship_length = ship_data
            horizontal = rotation_or_horizontal
            if horizontal:
                for i in range(ship_length):
                    cells.add((x + i, y))
            else:
                for i in range(ship_length):
                    cells.add((x, y + i))
        else:
            # Custom shaped ship
            cell_offsets = ship_data
            rotation_idx = rotation_or_horizontal
            
            # Apply rotation
            rotated = cell_offsets
            for _ in range(rotation_idx):
                rotated = [(dy, -dx) for dx, dy in rotated]
                min_x = min(rx for rx, ry in rotated)
                min_y = min(ry for rx, ry in rotated)
                rotated = [(rx - min_x, ry - min_y) for rx, ry in rotated]
            
            # Add to board position
            for dx, dy in rotated:
                cells.add((x + dx, y + dy))
        
        return cells
    
    def encode_state(self, state):
        """
        Encode a state into a compact numpy array for memory efficiency.
        
        Args:
            state: tuple of (ship_idx, x, y, horizontal)
            
        Returns:
            numpy array of uint8 values
        """
        import numpy as np
        # Each ship: [ship_idx, x, y, horizontal] = 4 bytes
        # Convert to flat array
        encoded = []
        for ship_idx, x, y, horizontal in state:
            encoded.extend([ship_idx, x, y, 1 if horizontal else 0])
        return np.array(encoded, dtype=np.uint8)
    
    def decode_state(self, encoded_state):
        """
        Decode a compact state back into the original format.
        
        Args:
            encoded_state: numpy array of uint8 values
            
        Returns:
            tuple of (ship_idx, x, y, horizontal)
        """
        state = []
        for i in range(0, len(encoded_state), 4):
            ship_idx = int(encoded_state[i])
            x = int(encoded_state[i + 1])
            y = int(encoded_state[i + 2])
            horizontal = bool(encoded_state[i + 3])
            state.append((ship_idx, x, y, horizontal))
        return tuple(state)
    
    def state_to_cells(self, state):
        """
        Convert a board state to all occupied cells.
        
        Args:
            state: tuple of (ship_idx, x, y, rotation_or_horizontal) for each ship
            
        Returns:
            set of (x, y) tuples
        """
        all_cells = set()
        for ship_idx, x, y, rotation_or_horizontal in state:
            ship_name, ship_data = self.ships[ship_idx]
            cells = self.placement_to_cells(x, y, rotation_or_horizontal, ship_data)
            all_cells.update(cells)
        return all_cells
    
    def is_valid_state(self, state):
        """
        Check if a board state is valid (no overlapping ships).
        
        Args:
            state: tuple of (ship_idx, x, y, rotation_or_horizontal) for each ship
            
        Returns:
            bool
        """
        all_cells = []
        for ship_idx, x, y, rotation_or_horizontal in state:
            ship_name, ship_data = self.ships[ship_idx]
            cells = self.placement_to_cells(x, y, rotation_or_horizontal, ship_data)
            all_cells.append(cells)
        
        # Check for overlaps
        for i in range(len(all_cells)):
            for j in range(i + 1, len(all_cells)):
                if all_cells[i] & all_cells[j]:  # Intersection
                    return False
        return True
    
    def generate_all_valid_states(self, checkpoint_interval=10_000_000):
        """
        Generate all valid board states (all ships placed, no overlaps).
        Uses smart recursive generation with pruning for efficiency.
        
        Args:
            checkpoint_interval: save progress every N states (0 to disable)
        """
        import time
        
        print("Generating all possible placements for each ship...")
        
        # Generate all possible placements for each ship
        all_ship_placements = []
        for ship_idx, (ship_name, ship_data) in enumerate(self.ships):
            # ship_data can be either an integer (length) or list of cell offsets
            if isinstance(ship_data, int):
                # Linear ship - use old method
                placements = self.generate_placements(ship_data)
            else:
                # Custom shape - generate rotated placements
                placements = self.generate_custom_placements(ship_data)
            
            # Tag each placement with its ship index
            tagged_placements = [(ship_idx, x, y, h) for x, y, h in placements]
            all_ship_placements.append(tagged_placements)
            
            ship_desc = f"length {ship_data}" if isinstance(ship_data, int) else f"custom shape"
            print(f"  {ship_name} ({ship_desc}): {len(placements)} possible placements")
        
        # Always use smart recursive generation
        print("Using smart recursive generation with pruning...")
        self._generate_smart(all_ship_placements, checkpoint_interval)
        
        return self.valid_states
    
    
    def _generate_smart(self, all_ship_placements, checkpoint_interval):
        """
        Smart generation: place ships one at a time, pruning invalid branches early.
        This is MUCH faster than checking all combinations.
        """
        import time
        
        print("Using smart recursive generation with pruning...")
        start_time = time.time()
        last_print_time = start_time
        last_checkpoint_count = 0
        
        def place_ships_recursive(ship_idx, occupied_cells, current_placement):
            """Recursively place ships, pruning conflicts early."""
            nonlocal last_print_time, last_checkpoint_count
            
            # Base case: all ships placed
            if ship_idx >= len(self.ships):
                self.valid_states.append(tuple(current_placement))
                
                # Progress updates - print every 5 seconds or every 1M states
                valid_count = len(self.valid_states)
                current_time = time.time()
                
                if (current_time - last_print_time >= 5.0) or (valid_count % 1_000_000 == 0):
                    elapsed = current_time - start_time
                    rate = valid_count / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {valid_count:,} valid states | "
                          f"Rate: {rate:,.0f}/s | Elapsed: {elapsed:.1f}s")
                    last_print_time = current_time
                
                # Checkpoint saving
                if checkpoint_interval > 0 and valid_count % checkpoint_interval == 0 and valid_count != last_checkpoint_count:
                    elapsed = current_time - start_time
                    print(f"  ✓ Checkpoint: {valid_count:,} states ({elapsed:.1f}s)")
                    self.save_states(self.get_cache_filename() + ".checkpoint")
                    last_checkpoint_count = valid_count
                
                return
            
            # Try each placement for current ship
            for placement in all_ship_placements[ship_idx]:
                _, x, y, rotation_or_horizontal = placement
                ship_name, ship_data = self.ships[ship_idx]
                ship_cells = self.placement_to_cells(x, y, rotation_or_horizontal, ship_data)
                
                # Check if this placement conflicts with already-placed ships
                if not (ship_cells & occupied_cells):  # No overlap
                    # Place this ship and recurse
                    new_occupied = occupied_cells | ship_cells
                    current_placement.append(placement)
                    place_ships_recursive(ship_idx + 1, new_occupied, current_placement)
                    current_placement.pop()
        
        # Start recursive generation
        place_ships_recursive(0, set(), [])
        
        elapsed = time.time() - start_time
        avg_rate = len(self.valid_states) / elapsed if elapsed > 0 else 0
        print(f"✓ Generated {len(self.valid_states):,} valid board states in {elapsed:.1f}s")
        print(f"  Average rate: {avg_rate:,.0f} states/second")
    
    def save_states(self, filename=None, compress=True):
        """
        Save valid states to a pickle file for later loading.
        
        Args:
            filename: optional custom filename, otherwise uses auto-generated name
            compress: if True, use gzip compression (slower but smaller files)
        """
        if not self.valid_states:
            print("Warning: No valid states to save!")
            return
        
        if filename is None:
            filename = self.get_cache_filename()
        
        print(f"Saving {len(self.valid_states):,} states to '{filename}'...")
        
        # Save both the states and the configuration
        cache_data = {
            'board_size': self.board_size,
            'ships': self.ships,
            'valid_states': self.valid_states
        }
        
        if compress:
            import gzip
            with gzip.open(filename + '.gz', 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            filename = filename + '.gz'
        else:
            with open(filename, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size = os.path.getsize(filename) / (1024 * 1024)  # Convert to MB
        print(f"✓ Saved to '{filename}' ({file_size:.2f} MB)")
    
    def load_states(self, filename=None):
        """
        Load valid states from a pickle file.
        
        Args:
            filename: optional custom filename, otherwise uses auto-generated name
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if filename is None:
            filename = self.get_cache_filename()
        
        # Try compressed version first
        if not os.path.exists(filename):
            if os.path.exists(filename + '.gz'):
                filename = filename + '.gz'
            else:
                print(f"Cache file '{filename}' not found.")
                return False
        
        print(f"Loading states from '{filename}'...")
        
        # Detect if file is compressed
        if filename.endswith('.gz'):
            import gzip
            with gzip.open(filename, 'rb') as f:
                cache_data = pickle.load(f)
        else:
            with open(filename, 'rb') as f:
                cache_data = pickle.load(f)
        
        # Verify the configuration matches
        if (cache_data['board_size'] != self.board_size or 
            cache_data['ships'] != self.ships):
            print("Warning: Cache file configuration doesn't match current setup!")
            print(f"  Cache: {cache_data['board_size']}x{cache_data['board_size']}, {cache_data['ships']}")
            print(f"  Current: {self.board_size}x{self.board_size}, {self.ships}")
            return False
        
        self.valid_states = cache_data['valid_states']
        print(f"✓ Loaded {len(self.valid_states):,} valid states")
        return True
    
    def calculate_probabilities(self, normalize=True):
        """
        Calculate the probability heatmap based on current valid states.
        
        Args:
            normalize: if True, scale values to 1-10 range
        
        Returns:
            2D list (board_size x board_size) with counts/probabilities
        """
        if not self.valid_states:
            print("Warning: No valid states generated yet!")
            return [[0] * self.board_size for _ in range(self.board_size)]
        
        # Count how many states have a ship at each cell
        cell_counts = defaultdict(int)
        
        for state in self.valid_states:
            occupied_cells = self.state_to_cells(state)
            for cell in occupied_cells:
                cell_counts[cell] += 1
        
        # Convert to 2D array (y, x indexing for easier printing)
        heatmap = [[0] * self.board_size for _ in range(self.board_size)]
        for (x, y), count in cell_counts.items():
            heatmap[y][x] = count
        
        # Set already-shot cells to 0 (they're not targetable)
        shot_cells = self.misses | self.hits
        for x, y in shot_cells:
            heatmap[y][x] = 0
        
        # Normalize to 1-10 scale if requested
        if normalize:
            # Only consider unshot cells for min/max
            unshot_values = []
            for y in range(self.board_size):
                for x in range(self.board_size):
                    if (x, y) not in shot_cells and heatmap[y][x] > 0:
                        unshot_values.append(heatmap[y][x])
            
            if unshot_values:  # Only normalize if there are unshot cells
                min_val = min(unshot_values)
                max_val = max(unshot_values)
                
                if max_val > min_val:  # Avoid division by zero
                    for y in range(self.board_size):
                        for x in range(self.board_size):
                            if (x, y) not in shot_cells and heatmap[y][x] > 0:
                                # Scale from [min_val, max_val] to [1, 10]
                                normalized = 1 + (heatmap[y][x] - min_val) * 9 / (max_val - min_val)
                                heatmap[y][x] = round(normalized)
        
        return heatmap
    
    def record_miss(self, x, y, allow_overwrite=False):
        """
        Record a miss at the given coordinates.
        
        Args:
            x, y: coordinates of the miss
            allow_overwrite: if True, allows changing a hit to a miss
        """
        if (x, y) in self.hits:
            if allow_overwrite:
                self.hits.discard((x, y))
                print(f"  (Changed from hit to miss)")
            else:
                print(f"Error: ({x}, {y}) was already a hit!")
                return
        
        self.misses.add((x, y))
        self.history.append(('miss', (x, y)))
    
    def record_hit(self, x, y, allow_overwrite=False):
        """
        Record a hit at the given coordinates.
        
        Args:
            x, y: coordinates of the hit
            allow_overwrite: if True, allows changing a miss to a hit
        """
        if (x, y) in self.misses:
            if allow_overwrite:
                self.misses.discard((x, y))
                print(f"  (Changed from miss to hit)")
            else:
                print(f"Error: ({x}, {y}) was already a miss!")
                return
        
        self.hits.add((x, y))
        self.history.append(('hit', (x, y)))
    
    def record_sunk_ship(self, ship_idx, hit_coords):
        """
        Record that a ship has been sunk.
        
        Args:
            ship_idx: index of the ship that was sunk (0, 1, 2, etc.)
            hit_coords: set or list of (x, y) coordinates where this ship was hit
        """
        # Validate ship index
        if ship_idx < 0 or ship_idx >= len(self.ships):
            print(f"Error: Invalid ship index {ship_idx}!")
            return
        
        # Verify all coords are in hits
        hit_coords_set = set(hit_coords)
        if not hit_coords_set.issubset(self.hits):
            print(f"Error: Some coordinates aren't marked as hits!")
            return
        
        # Verify correct number of hits for this ship
        ship_name, ship_data = self.ships[ship_idx]
        if isinstance(ship_data, int):
            expected_hits = ship_data
        else:
            expected_hits = len(ship_data)
        
        if len(hit_coords_set) != expected_hits:
            print(f"Error: {ship_name} should have {expected_hits} hits, but {len(hit_coords_set)} coords provided!")
            return
        
        self.sunk_ships.append((ship_idx, hit_coords_set))
        self.history.append(('sunk', (ship_idx, hit_coords_set)))
        print(f"✓ {ship_name} sunk at {sorted(hit_coords_set)}")
    
    def undo(self):
        """
        Undo the last action.
        
        Returns:
            bool: True if undo was successful, False if nothing to undo
        """
        if not self.history:
            print("Nothing to undo!")
            return False
        
        action_type, data = self.history.pop()
        
        if action_type == 'miss':
            x, y = data
            self.misses.discard((x, y))
            print(f"✓ Undid miss at ({x}, {y})")
        elif action_type == 'hit':
            x, y = data
            self.hits.discard((x, y))
            print(f"✓ Undid hit at ({x}, {y})")
        elif action_type == 'sunk':
            ship_idx, coords = data
            # Remove the last sunk ship entry
            self.sunk_ships = [(idx, c) for idx, c in self.sunk_ships if not (idx == ship_idx and c == coords)]
            ship_name, _ = self.ships[ship_idx]
            print(f"✓ Undid sinking of {ship_name}")
        
        return True
    
    def filter_states_by_misses(self, states=None):
        """
        Filter board states to remove those that have ships on known miss locations.
        
        Args:
            states: list of states to filter (uses self.valid_states if None)
            
        Returns:
            filtered list of states
        """
        if states is None:
            states = self.valid_states
        
        if not self.misses:
            return states
        
        filtered = []
        for state in states:
            occupied_cells = self.state_to_cells(state)
            # If any miss location has a ship in this state, reject it
            if not (self.misses & occupied_cells):  # No intersection
                filtered.append(state)
        
        return filtered
    
    def filter_states_by_hits(self, states=None):
        """
        Filter board states to keep only those that have ships covering all known hits.
        
        Args:
            states: list of states to filter (uses self.valid_states if None)
            
        Returns:
            filtered list of states
        """
        if states is None:
            states = self.valid_states
        
        if not self.hits:
            return states
        
        filtered = []
        for state in states:
            occupied_cells = self.state_to_cells(state)
            # All hit locations must be covered by ships in this state
            if self.hits.issubset(occupied_cells):
                filtered.append(state)
        
        return filtered
    
    def filter_states_by_sunk_ships(self, states=None):
        """
        Filter board states to match sunk ship constraints.
        
        A state is valid only if:
        - Each sunk ship occupies exactly its recorded hit positions
        - The ship type matches
        
        Args:
            states: list of states to filter (uses self.valid_states if None)
            
        Returns:
            filtered list of states
        """
        if states is None:
            states = self.valid_states
        
        if not self.sunk_ships:
            return states
        
        filtered = []
        for state in states:
            valid = True
            
            # Check each sunk ship
            for sunk_ship_idx, sunk_coords in self.sunk_ships:
                # Find this ship in the current state
                found = False
                for ship_idx, x, y, rotation_or_horizontal in state:
                    if ship_idx == sunk_ship_idx:
                        # Get cells this ship occupies in this state
                        ship_name, ship_data = self.ships[ship_idx]
                        ship_cells = self.placement_to_cells(x, y, rotation_or_horizontal, ship_data)
                        
                        # Must exactly match the sunk coordinates
                        if ship_cells == sunk_coords:
                            found = True
                            break
                
                if not found:
                    valid = False
                    break
            
            if valid:
                filtered.append(state)
        
        return filtered
    
    def get_current_valid_states(self):
        """
        Get the current set of valid states after applying all filters.
        
        Returns:
            filtered list of states
        """
        # Start with all valid states
        current_states = self.valid_states
        
        # Apply filters in order
        current_states = self.filter_states_by_misses(current_states)
        current_states = self.filter_states_by_hits(current_states)
        current_states = self.filter_states_by_sunk_ships(current_states)
        
        return current_states
    
    def get_current_probabilities(self, normalize=True):
        """
        Calculate probabilities based on current game state (after filtering).
        
        Args:
            normalize: if True, scale values to 1-10 range
            
        Returns:
            2D heatmap array
        """
        current_states = self.get_current_valid_states()
        
        if not current_states:
            print("Warning: No valid states remain! Something went wrong.")
            return [[0] * self.board_size for _ in range(self.board_size)]
        
        # Temporarily swap valid_states to calculate probabilities
        original_states = self.valid_states
        self.valid_states = current_states
        heatmap = self.calculate_probabilities(normalize)
        self.valid_states = original_states
        
        return heatmap
    
    def print_heatmap(self, heatmap, normalized=True):
        """Pretty print the probability heatmap with hits/misses marked."""
        if normalized:
            print("\nProbability Heatmap (1=low, 10=high, X=miss, H=hit):")
        else:
            print("\nProbability Heatmap (raw counts, X=miss, H=hit):")
        
        # Column headers (letters)
        col_letters = [chr(ord('A') + i) for i in range(self.board_size)]
        print("   |" + "".join(f"    {letter}" for letter in col_letters))
        print("=" * (self.board_size * 6 + 5))
        
        # Print rows from bottom to top (reversed), with 1-indexed row numbers
        for y in range(self.board_size - 1, -1, -1):
            row_num = y + 1  # Convert to 1-indexed
            row = heatmap[y]
            print(f"{row_num:2d} |", end="")
            for x, val in enumerate(row):
                if (x, y) in self.misses:
                    print(f"    X", end=" ")
                elif (x, y) in self.hits:
                    print(f"    H", end=" ")
                else:
                    print(f"{val:5d}", end=" ")
            print()
        
        print("=" * (self.board_size * 6 + 5))


def interactive_loop(engine):
    """
    Simple interactive loop for testing the engine.
    
    Commands:
        1 - Add a hit
        0 - Add a miss
        s - Mark a ship as sunk
        u - Undo last action
        q - Quit
    """
    print("\n" + "=" * 50)
    print("INTERACTIVE MODE")
    print("=" * 50)
    print("Commands: 1=hit, 0=miss, s=sink ship, u=undo, q=quit")
    print("Coordinates: Column=letter (A,B,C...), Row=number (1,2,3...)")
    
    def parse_coord(coord_str):
        """Parse coordinate like 'A1' or 'a1' into (x, y) 0-indexed."""
        coord_str = coord_str.strip().upper()
        if len(coord_str) < 2:
            raise ValueError("Coordinate too short")
        
        col_letter = coord_str[0]
        row_str = coord_str[1:]
        
        x = ord(col_letter) - ord('A')
        y = int(row_str) - 1  # Convert from 1-indexed to 0-indexed
        
        return x, y
    
    while True:
        # Show current board
        heatmap = engine.get_current_probabilities()
        engine.print_heatmap(heatmap)
        
        # Get valid states count
        valid_count = len(engine.get_current_valid_states())
        print(f"\nValid board states: {valid_count:,}")
        
        # Show sunk ships
        if engine.sunk_ships:
            print("Sunk ships:", end=" ")
            for ship_idx, coords in engine.sunk_ships:
                ship_name, _ = engine.ships[ship_idx]
                print(f"{ship_name}", end=" ")
            print()
        
        # Get command
        cmd = input("\nEnter command (1/0/s/u/q): ").strip().lower()
        
        if cmd == 'q':
            print("Exiting...")
            break
        elif cmd == 'u':
            engine.undo()
        elif cmd == '1':
            try:
                coord = input("  Enter coordinate (e.g., A1, B3): ")
                x, y = parse_coord(coord)
                if 0 <= x < engine.board_size and 0 <= y < engine.board_size:
                    engine.record_hit(x, y, allow_overwrite=True)
                    print(f"✓ Hit recorded at {coord.upper()}")
                else:
                    print(f"Invalid coordinates! Must be A-{chr(ord('A') + engine.board_size - 1)}, 1-{engine.board_size}")
            except (ValueError, IndexError):
                print("Invalid input! Use format like A1, B3, etc.")
        elif cmd == '0':
            try:
                coord = input("  Enter coordinate (e.g., A1, B3): ")
                x, y = parse_coord(coord)
                if 0 <= x < engine.board_size and 0 <= y < engine.board_size:
                    engine.record_miss(x, y, allow_overwrite=True)
                    print(f"✓ Miss recorded at {coord.upper()}")
                else:
                    print(f"Invalid coordinates! Must be A-{chr(ord('A') + engine.board_size - 1)}, 1-{engine.board_size}")
            except (ValueError, IndexError):
                print("Invalid input! Use format like A1, B3, etc.")
        elif cmd == 's':
            # Show available ships
            print("\nAvailable ships:")
            for idx, (name, ship_data) in enumerate(engine.ships):
                already_sunk = any(s_idx == idx for s_idx, _ in engine.sunk_ships)
                status = "(SUNK)" if already_sunk else ""
                if isinstance(ship_data, int):
                    ship_desc = f"length {ship_data}"
                else:
                    ship_desc = f"{len(ship_data)} cells, custom shape"
                print(f"  {idx}: {name} ({ship_desc}) {status}")
            
            try:
                ship_idx = int(input("  Enter ship number: "))
                coords_input = input("  Enter hit coordinates (e.g., 'A1 B1 C1'): ").strip()
                
                # Parse coordinates
                coords = []
                for coord_str in coords_input.split():
                    x, y = parse_coord(coord_str)
                    coords.append((x, y))
                
                engine.record_sunk_ship(ship_idx, coords)
            except (ValueError, IndexError):
                print("Invalid input! Use format like: A1 B1 C1")
        else:
            print("Unknown command! Use 1 (hit), 0 (miss), s (sink), u (undo), or q (quit)")
        
        print()  # Blank line for readability


def main():
    """Run the interactive battleship engine."""
    print("=" * 50)
    print("BATTLESHIP PROBABILITY ENGINE")
    print("=" * 50)
    
    # Standard 10x10 Battleship
    engine = BattleshipEngine( #More than 4 ships NOT RECCOMENDED (very slow and takes high memory)
        board_size=10,
        ships=[
            ("Carrier", 5),
            ("Battleship", 4),
            #("Cruiser", 3),
            ("LShape", [(0,0), (1,0), (0,1)]),
            #("Submarine", 3),
            #("Destroyer", 2)
        ]
    )
    # Load or generate states
    if not engine.load_states():
        print("Generating states (this may take a while for larger boards)...")
        engine.generate_all_valid_states()
        engine.save_states()
    
    # Start interactive loop
    interactive_loop(engine)


if __name__ == "__main__":
    main()