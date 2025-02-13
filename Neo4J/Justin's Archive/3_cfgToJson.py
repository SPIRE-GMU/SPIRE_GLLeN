# Script to extract data from multiple CFG files and assign unique identifiers for Neo4j database integration.
import re
import uuid
import os
import json

# Initialize an empty dictionary to hold extracted CFG data
cfg_data = {}

# Define a function to extract elements from CFG files
def extract_cfg_elements(file_content, function_prefix):
    data = {
        'function_node': {},
        'basic_blocks': {},
        'loops': [],
        'decision_nodes': [],
        'edges': []
    }

    # Assign a unique identifier to the function
    function_id = f"{function_prefix}-func-{str(uuid.uuid4())}"
    data['function_node'] = {
        'function_id': function_id,
        'function_name': None,
        'full_code': "",
    }

    current_loop = None
    current_block = None
    in_function_code = True
    decision_pending = None

    for line in file_content:
        line = line.strip()
        data['function_node']['full_code'] += line + "\n"

        # Extract function name and metadata
        if line.startswith(";; Function"):
            match = re.match(r";; Function (\w+)", line)
            if match:
                data['function_node']['function_name'] = match.group(1)
                function_name = match.group(1)
                function_prefix = f"{function_name}-{str(uuid.uuid4())[:8]}"
                data['function_node']['function_id'] = function_prefix

        # Detect loop headers
        elif line.startswith(";; Loop"):
            match = re.match(r";; Loop (\d+)", line)
            if match:
                current_loop = {
                    'loop_id': f"{function_prefix}-loop-{match.group(1)}",
                    'header': None,
                    'latch': None,
                    'depth': None,
                    'nodes': []
                }
                data['loops'].append(current_loop)

        # Extract loop properties
        elif line.startswith(";;  header"):
            if current_loop:
                current_loop['header'] = int(re.search(r"header (\d+)", line).group(1))
        elif line.startswith(";;  latch"):
            if current_loop:
                current_loop['latch'] = int(re.search(r"latch (\d+)", line).group(1))
        elif line.startswith(";;  depth"):
            if current_loop:
                current_loop['depth'] = int(re.search(r"depth (\d+)", line).group(1))
        elif line.startswith(";;  nodes"):
            if current_loop:
                nodes = [int(n) for n in re.findall(r"\d+", line)]
                current_loop['nodes'].extend(nodes)

        # Detect basic block start
        elif line.startswith("<bb"):
            match = re.match(r"<bb (\d+)>", line)
            if match:
                block_id = int(match.group(1))
                block_uuid = f"{function_prefix}-bb-{block_id}"
                current_block = {
                    'block_id': block_uuid,
                    'block_code': f"<bb {block_id}> :\n",
                    'instructions': [],
                    'contains_decision': False,
                    'is_loop_header': False,
                    'successors': []
                }
                data['basic_blocks'][block_uuid] = current_block
                
                # Mark block as a loop header if it matches any loop header
                if current_loop and current_loop['header'] == block_id:
                    current_block['is_loop_header'] = True

        # Extract basic block instructions and identify if statements
        elif current_block and not line.startswith(";;"):
            current_block['block_code'] += line + "\n"
            if "if" in line:
                decision_uuid = f"{function_prefix}-decision-{str(uuid.uuid4())[:8]}"
                decision = {
                    'decision_id': decision_uuid,
                    'block_id': current_block['block_id'],
                    'condition': line,
                    'true_target': None,
                    'false_target': None
                }
                current_block['contains_decision'] = True
                data['decision_nodes'].append(decision)
                decision_pending = decision  # Keep track of the decision to set targets later
            current_block['instructions'].append(line)

        # Extract successors (edges between nodes)
        elif line.startswith(";; ") and "succs" in line:
            match = re.match(r";; (\d+) succs \{ ([\d\s]+)\ }", line)
            if match:
                src_block = f"{function_prefix}-bb-{match.group(1)}"
                successors = [int(s) for s in match.group(2).split()]
                for succ in successors:
                    succ_block = f"{function_prefix}-bb-{succ}"
                    data['edges'].append({'from': src_block, 'to': succ_block})
                    if src_block in data['basic_blocks']:
                        data['basic_blocks'][src_block]['successors'].append(succ_block)
                    # Set true/false targets for decision nodes if applicable
                    if decision_pending and decision_pending['block_id'] == src_block:
                        if decision_pending['true_target'] is None:
                            decision_pending['true_target'] = succ_block
                        elif decision_pending['false_target'] is None:
                            decision_pending['false_target'] = succ_block
                
                # Clear decision_pending once targets are set
                if decision_pending and decision_pending['true_target'] and decision_pending['false_target']:
                    decision_pending = None

    # Assign latches as successors where applicable for loops
    for loop in data['loops']:
        if loop['latch'] is not None:
            latch_block_uuid = f"{function_prefix}-bb-{loop['latch']}"
            header_block_uuid = f"{function_prefix}-bb-{loop['header']}"
            if latch_block_uuid in data['basic_blocks'] and header_block_uuid in data['basic_blocks']:
                data['edges'].append({'from': latch_block_uuid, 'to': header_block_uuid})
                data['basic_blocks'][latch_block_uuid]['successors'].append(header_block_uuid)

    # Ensure decision node targets are correctly updated
    for decision in data['decision_nodes']:
        if decision['true_target'] is None or decision['false_target'] is None:
            block_id = decision['block_id']
            if block_id in data['basic_blocks']:
                successors = data['basic_blocks'][block_id]['successors']
                if len(successors) >= 2:
                    decision['true_target'] = successors[0]
                    decision['false_target'] = successors[1]
                elif len(successors) == 1:
                    decision['true_target'] = successors[0]

    return data

# Process each CFG file and extract relevant elements
cfg_file_paths = []

# Walk through the directory to find all CFG files in the specified folder
cfg_directory = '/home/spire2/SPIRE_GLLeN/Neo4J/cfg_files'
for root, dirs, files in os.walk(cfg_directory):
    for file in files:
        if file.endswith('.cfg'):
            cfg_file_paths.append(os.path.join(root, file))

for file_path in cfg_file_paths:
    with open(file_path, 'r') as file:
        file_content = file.readlines()
        function_prefix = str(uuid.uuid4())[:8]  # Generate a unique prefix for each CFG
        cfg_data[file_path] = extract_cfg_elements(file_content, function_prefix)

# Write the extracted CFG data to a JSON file for easy inspection
output_file_path = '/home/spire2/SPIRE_GLLeN/Neo4J/cfg_data.json'
with open(output_file_path, 'w') as json_file:
    json.dump(cfg_data, json_file, indent=4)

print(f"CFG data has been written to {output_file_path}")
