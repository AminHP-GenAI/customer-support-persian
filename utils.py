from typing import List, Dict


def list_of_dict_to_str(lod: List[Dict]) -> str:
    result = ""

    for i, dic in enumerate(lod):
        result += f"Item {i+1}:\n"
        for k, v in dic.items():
            result += f"-- {k}: {v}\n"

    return result
