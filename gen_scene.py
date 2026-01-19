import argparse
import random
import sys

# --- Configuration ---
# Standard Sci-Fi Material Palette (r, g, b, diff, spec, shin)
PALETTE = {
    'concrete': (0.3, 0.3, 0.35, 0.9, 0.1, 10),
    'glass':    (0.1, 0.2, 0.5, 0.5, 0.9, 80),
    'light':    (0.9, 0.9, 0.6, 1.0, 1.0, 100),
    'steel':    (0.5, 0.5, 0.55, 0.6, 0.8, 50),
    'grass':    (0.1, 0.4, 0.1, 0.8, 0.0, 1),
}

class Node:
    def __init__(self):
        self.count = 0 

    def serialize(self, indent=0):
        raise NotImplementedError

class Primitive(Node):
    def __init__(self, type_name, params):
        super().__init__()
        self.type_name = type_name
        self.params = params
        self.count = 1

    def serialize(self, indent=0):
        s = " " * indent + f"{self.type_name} "
        s += " ".join(f"{x:.4f}" for x in self.params)
        return s

class Operation(Node):
    def __init__(self, op_name, left, right):
        super().__init__()
        self.op_name = op_name
        self.left = left
        self.right = right
        self.count = 1 + left.count + right.count

    def serialize(self, indent=0):
        s = " " * indent + f"{self.op_name}\n"
        s += self.left.serialize(indent + 2) + "\n"
        s += self.right.serialize(indent + 2)
        return s

# --- Generators ---

def create_cuboid(x, y, z, w, h, d, mat):
    params = [x, y, z, w, h, d, *mat]
    return Primitive("cuboid", params)

def create_cylinder(x, y, z, r, h, mat):
    params = [x, y, z, r, h, *mat]
    return Primitive("cylinder", params)

def create_sphere(x, y, z, rad, mat):
    params = [x, y, z, rad, *mat]
    return Primitive("sphere", params)

def create_cone(x, y, z, rad, h, mat):
    params = [x, y, z, rad, h, *mat]
    return Primitive("cone", params)

def generate_building(bx, bz):
    width = random.uniform(2.0, 4.0)
    depth = random.uniform(2.0, 4.0)
    height = random.uniform(5.0, 15.0)
    
    y_center = height / 2.0
    
    base = create_cuboid(bx, y_center, bz, width, height, depth, PALETTE['concrete'])
    current_tree = base
    
    # Add Detail: Spire or Second Tier
    if random.random() > 0.3:
        tier_h = random.uniform(2.0, 6.0)
        tier_w = width * 0.6
        tier_d = depth * 0.6
        tier_y = height + (tier_h / 2.0)
        
        tier = create_cuboid(bx, tier_y, bz, tier_w, tier_h, tier_d, PALETTE['glass'])
        current_tree = Operation("union", current_tree, tier)
        
        # Add Antenna
        if random.random() > 0.5:
            ant_h = random.uniform(1.0, 4.0)
            ant_r = 0.2
            ant_y = height + tier_h + (ant_h / 2.0) - 0.5
            
            antenna = create_cylinder(bx, ant_y, bz, ant_r, ant_h, PALETTE['steel'])
            current_tree = Operation("union", current_tree, antenna)

    # Add Entrance (Subtraction)
    door_h = 1.5
    door_w = 1.2
    door_d = depth + 1.0 
    door_y = door_h / 2.0
    
    door = create_cuboid(bx, door_y, bz, door_w, door_h, door_d, PALETTE['concrete'])
    current_tree = Operation("difference", current_tree, door)
    
    return current_tree

def balance_tree(objects):
    """
    Combines objects into a balanced binary tree to prevent recursion overflow.
    """
    if not objects:
        return None
    if len(objects) == 1:
        return objects[0]
    
    mid = len(objects) // 2
    left = balance_tree(objects[:mid])
    right = balance_tree(objects[mid:])
    
    return Operation("union", left, right)

def generate_scene(target_nodes):
    objects = []
    
    # Create Floor
    floor_size = int(target_nodes ** 0.5) * 2 
    if floor_size < 20: floor_size = 20
    floor = create_cuboid(0, -1, 0, floor_size, 1, floor_size, PALETTE['grass'])
    objects.append(floor)
    
    node_count = 1
    grid_step = 8.0
    x, z = -floor_size/2 + 4, -floor_size/2 + 4
    
    print(f"Generating scene with approx {target_nodes} nodes...", file=sys.stdout)
    
    while node_count < target_nodes:
        building = generate_building(x, z)
        objects.append(building)
        node_count += building.count + 1 
        
        x += grid_step + random.uniform(-1, 1)
        if x > floor_size/2 - 4:
            x = -floor_size/2 + 4
            z += grid_step + random.uniform(-1, 1)
            
        if z > floor_size/2 - 4:
            break

    return balance_tree(objects)

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Generate a procedural CSG scene.")
    parser.add_argument("nodes", type=int, help="Target number of nodes to generate")
    parser.add_argument("output_file", type=str, help="Output filename (e.g., scene.txt)")
    args = parser.parse_args()

    if args.nodes < 1:
        print("Error: Nodes must be positive.")
        sys.exit(1)

    scene = generate_scene(args.nodes)
    
    # Write directly to file using UTF-8 to avoid BOM/Encoding issues
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(scene.serialize())
        print(f"Success! Written {scene.count} nodes to '{args.output_file}'.")
    except Exception as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    main()