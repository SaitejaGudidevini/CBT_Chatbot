#!/usr/bin/env python3
import json
import os
import sys

def process_conversation(conversation):
    """
    Process a single conversation into three progressive versions:
    1. System + first patient turn + first therapist turn
    2. System + first patient/therapist exchange + second patient/therapist exchange
    3. Full conversation minus the final patient response
    """
    # Ensure the conversation has the expected structure
    if len(conversation) < 5:
        print(f"Warning: Conversation too short, skipping: {conversation}")
        return []
    
    system_message = conversation[0]  # Extract system message
    
    # Create the three versions
    version1 = [system_message, conversation[1], conversation[2]]
    
    # For version 2, check if we have enough messages
    if len(conversation) >= 5:
        version2 = [system_message, conversation[1], conversation[2], conversation[3], conversation[4]]
    else:
        version2 = None
    
    # For version 3, include all except final patient response (if it exists)
    if len(conversation) % 2 == 0:  # Even number of messages means last one is patient
        version3 = conversation[:-1]
    else:
        version3 = conversation  # Odd number means last one is therapist, keep all
    
    # Return all valid versions
    result = [version1]
    if version2:
        result.append(version2)
    if version3 and len(version3) > len(version1):  # Only add version3 if it's different from version1
        result.append(version3)
    
    return result

def main():
    # Determine input file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'cbt_finetuning_dataset.json')
    output_file = os.path.join(script_dir, 'cbt_finetuning_dataset_littleconvo.json')
    
    # Load the input data
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Input file '{input_file}' is not valid JSON.")
        sys.exit(1)
    
    # Process each conversation
    processed_data = []
    for conversation in data:
        processed_data.extend(process_conversation(conversation))
    
    # Write the output
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Processed {len(data)} conversations into {len(processed_data)} smaller conversations.")
    print(f"Output written to {output_file}")

if __name__ == "__main__":
    main()
