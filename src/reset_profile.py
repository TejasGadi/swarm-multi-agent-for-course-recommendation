from pathlib import Path
import json

profile_json_path = Path("./profile.json")

def reset_profile_to_empty():
    if not profile_json_path.exists():
        print("profile.json not found.")
        return

    with profile_json_path.open("r", encoding="utf-8") as f:
        profile = json.load(f)

    def empty_value(val):
        if isinstance(val, str):
            return ""
        elif isinstance(val, bool):
            return False
        elif isinstance(val, int) or isinstance(val, float):
            return None
        elif isinstance(val, list):
            return []
        elif val is None:
            return None
        else:
            return None

    for key in profile.keys():
        profile[key] = empty_value(profile[key])

    with profile_json_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print("Profile reset to empty values.")