import json

def clean_data_from_file(filename="data.json"):
    """
    Reads data from a JSON file, cleans it, and writes the cleaned data back to the file.
    Enforces the 5-key structure and duration constraints.
    Includes sanity check print statements.

    Args:
        filename (str): The name of the JSON file to process.
    """

    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{filename}'.")
        return

    cleaned_data = clean_data(data)

    # Sanity Check: Print some summary info before writing back
    print("\n--- Data Cleaning Summary ---")
    total_sections = len(cleaned_data)
    total_items = 0
    for section_data in cleaned_data.values():
        total_items += len(section_data.get('items', []))
    print(f"Total Sections: {total_sections}")
    print(f"Total Items: {total_items}")
    print("---------------------------\n")

    try:
        with open(filename, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        print(f"Data cleaned and written back to '{filename}'")
    except IOError:
        print(f"Error: Could not write cleaned data to '{filename}'.")


def clean_data(data):
    """
    Cleans the input data according to the specified rules.
    Enforces that each item has only the keys: "content", "level", "duration", "reason", "remarks".
    Ensures duration is positive and either 25 or 50.
    Includes detailed sanity check print statements during cleaning.

    Args:
        data (dict): A dictionary where keys are section numbers and
                   values are dictionaries containing 'name' and 'items'.

    Returns:
        dict: A cleaned version of the input data.
    """

    required_keys = ["content", "level", "duration", "reason", "remarks"]
    cleaned_sections = 0
    cleaned_items = 0
    duration_values = set()  # To store unique duration values

    for section_num, section_data in data.items():
        if 'name' not in section_data or 'items' not in section_data:
            print(f"Warning: Section {section_num} is missing 'name' or 'items'. Skipping.")
            continue

        cleaned_sections += 1
        for item in section_data['items']:
            cleaned_items += 1
            cleaned_item = {}
            for key in required_keys:
                cleaned_item[key] = item.get(key)
            section_data['items'][section_data['items'].index(item)] = cleaned_item

            # Clean 'level'
            if 'level' in item and item['level']:
                original_level = item['level']
                level = item['level'].upper()
                if 'OR' in level:
                    level = level.split('OR')[-1].strip()
                item['level'] = level
                if len(item['level']) > 2:
                    item['level'] = item['level'][:2]
                if original_level != item['level']:
                    print(f"  - Cleaned level in Section {section_num}: '{original_level}' -> '{item['level']}'")

            # Clean 'duration'
            if 'duration' in item and item['duration'] is not None:
                original_duration = item['duration']
                if not isinstance(item['duration'], int) or item['duration'] < 0 or item['duration'] not in [25, 50]:
                    print(f"  - Invalid duration in Section {section_num}, item '{item.get('content', '')}': '{original_duration}'. Setting to None.")
                    item['duration'] = None
                elif original_duration != item['duration']:
                    print(f"  - Cleaned duration in Section {section_num}: '{original_duration}' -> '{item['duration']}'")
                else:
                    duration_values.add(item['duration'])  # Add valid duration to the set

    print(f"\nCleaned {cleaned_sections} sections and {cleaned_items} items.")
    print(f"Unique duration values: {duration_values}")  # Print the unique duration values
    return data
# Example Usage
clean_data_from_file("data.json")  # Cleans the file named "data.json"