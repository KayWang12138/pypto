import re

opcode_h_path = '/home/czq/code/pypto/framework/src/interface/operation/opcode.h'
isa_h_path = '/home/czq/code/pypto/framework/src/cost_model/simulation/common/ISA.h'

# 1. Parse opcode.h to get the ordered list of ops and comments
ordered_ops = [] # List of (type, value). type='op' or 'comment'
with open(opcode_h_path, 'r') as f:
    lines = f.readlines()

in_enum = False
for line in lines:
    line = line.strip()
    if 'enum class Opcode' in line:
        in_enum = True
        continue
    if in_enum and '};' in line:
        break
    if in_enum:
        if line.startswith('//'):
            ordered_ops.append(('comment', line))
        elif line.startswith('OP_'):
            # Extract op name, remove trailing comma
            op_part = line.split(',')[0].strip()
            if op_part.startswith('OP_'):
                op_name = op_part[3:]
                ordered_ops.append(('op', op_name))

# 2. Parse ISA.h to get current mappings
isa_map = {}
with open(isa_h_path, 'r') as f:
    isa_content = f.read()

# Regex to find map entries: {"NAME", CorePipeType::TYPE},
# We use findall to capture all.
# Note: This regex assumes the format in ISA.h is fairly consistent.
pattern = re.compile(r'{"(\w+)",\s*(CorePipeType::\w+)\}')
matches = pattern.findall(isa_content)

for name, pipe_type in matches:
    isa_map[name] = pipe_type

# 3. Generate new content for the map
new_map_lines = []
new_map_lines.append("const std::map<std::string, CorePipeType> SCHED_CORE_PIPE_TYPE {")

# Keep track of used ops to identify if any from ISA.h were missed (orphan check)
used_ops = set()

# Helper to check if we just added a comment
last_was_comment = False

for item_type, item_value in ordered_ops:
    if item_type == 'comment':
        # Ensure indentation
        new_map_lines.append(f"    {item_value}")
        last_was_comment = True
    elif item_type == 'op':
        op_name = item_value
        if op_name in isa_map:
            new_map_lines.append(f"    {{\"{op_name}\", {isa_map[op_name]}}}, ")
            used_ops.add(op_name)
            last_was_comment = False
        else:
            # Op in opcode.h but not in ISA.h. Skip.
            pass

# Add any ops from ISA.h that weren't in opcode.h
leftovers = []
for op_name, pipe_type in isa_map.items():
    if op_name not in used_ops:
        leftovers.append((op_name, pipe_type))

if leftovers:
    new_map_lines.append("    // Ops present in ISA.h but not found in opcode.h")
    for op_name, pipe_type in leftovers:
        new_map_lines.append(f"    {{\"{op_name}\", {pipe_type}}}, ")

new_map_lines.append("};")

# 4. Replace the map in ISA.h
start_marker = "const std::map<std::string, CorePipeType> SCHED_CORE_PIPE_TYPE {"
end_marker = "};"

start_idx = isa_content.find(start_marker)
# Find the first }; after start_marker
end_idx = isa_content.find(end_marker, start_idx)

if start_idx != -1 and end_idx != -1:
    # We need to capture the text up to start_marker (inclusive of "TYPE {")? 
    # Actually start_marker includes the brace.
    # But wait, the original file has comments on the same line as the opening brace: "TYPE { // Unary Vector"
    # My script will put comments on the next line. This is acceptable.
    
    # We want to replace from start_marker to end_marker + len(end_marker)
    
    new_content = isa_content[:start_idx] + "\n".join(new_map_lines) + isa_content[end_idx+len(end_marker):]
    
    # Write to file
    with open(isa_h_path, 'w') as f:
        f.write(new_content)
    print("Successfully updated ISA.h")
else:
    print("Could not find SCHED_CORE_PIPE_TYPE block in ISA.h")
