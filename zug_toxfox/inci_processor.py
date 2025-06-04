import json
import os
import re

import openpyxl

from zug_toxfox import default_config


class ProcessInci:
    def __init__(
        self, inci_xlsx_path: str, existing_json_path: str | None = None, output_path: str = "data/inci/inci.json"
    ):
        self.inci_xlsx_path = inci_xlsx_path
        self.existing_json_path = existing_json_path
        self.output_path = output_path

        self.endings = sorted(
            [
                "extract",
                "extract ferment filtrate",
                "ferment filtrate",
                "powder",
                "meristem cell culture conditioned media",
                "meristem cell culture",
                "meristem cell culture extract",
                "water",
                "ferment",
                "extract filtrate",
                "filtrate",
                "ferment extract",
                "extract ferment lysate",
                "ferment lysate extract filtrate",
                "lysate extract",
                "ferment extract filtrate",
                "ferment lysate extract",
                "lysate filtrate extract",
                "ferment filtrate extract",
                "fluoride hydroxyapatite",
            ],
            key=len,
            reverse=True,
        )

        self.substances = [
            "bark",
            "bulb",
            "pollen",
            "cap",
            "stem",
            "cell",
            "rice",
            "branchlet",
            "branch",
            "peel",
            "rhizome",
            "seed",
            "leaf",
            "bud",
            "root",
            "root juice",
            "flower",
            "stalk",
            "leaf oil",
            "stem juice",
            "stem oil",
            "sprout",
            "seed oil",
            "root oil",
            "rhizome",
            "leaf cell",
            "skin",
            "tricaprylate",
            "tricaprate",
            "scale",
            "sap",
        ]

        self.prefixes = ["va", "vp", "tdi"]

    def load_existing_json(self, json_file: str) -> set[str]:
        """Load existing .json file and return data as a set."""
        try:
            with open(json_file, encoding="utf-8") as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            existing_data = []
        return set(existing_data)

    def xlsx_to_list(self, simple: bool = False) -> set[str]:
        """Convert .xlsx file to a set of INCI entries."""
        entries = []

        wb = openpyxl.load_workbook(self.inci_xlsx_path)
        sheet = wb.active

        for row in sheet.iter_rows(min_row=2, min_col=2, max_col=2, values_only=True):
            column = str(row[0]).strip().lower()
            if simple:
                entries.append(column)
            else:
                entries.extend(self.get_split(column))

        return {entry for entry in entries if len(entry) > 2}

    def clean_entry(self, entry: str) -> str:
        """Clean the entry by removing unwanted characters and patterns."""
        if "(" in entry and ")" not in entry:
            entry = entry.replace("(", "")
        elif ")" in entry and "(" not in entry:
            entry = entry.replace(")", "")

        entry = entry.replace(" &", "/")
        entry = entry.replace("'", "")
        return entry.lower()

    def split_outside_parentheses(self, entry: str) -> list[str]:
        """Split a string at slashes that are outside of parentheses."""
        entry_list = self.split_prefix(entry)
        parenthesis_pattern = re.compile(r"/(?=(?:[^()]*\([^()]*\))*[^()]*$)")
        parts = []
        for s in entry_list:
            split_positions = [m.start() for m in parenthesis_pattern.finditer(s)]
            last_pos = 0
            for pos in split_positions:
                parts.append(s[last_pos:pos])
                last_pos = pos + 1
            parts.append(s[last_pos:])
        return parts

    def get_substance(self, entry: str) -> tuple[str, list[str]]:
        """Get substance components from an entry."""
        pattern = re.compile(
            r"\b("
            + "|".join(map(re.escape, self.substances))
            + r")(?:/(?:\b("
            + "|".join(map(re.escape, self.substances))
            + r")))*$"
        )
        match = pattern.search(entry)
        if match:
            matched_substances = match.group(0).split("/")
            if len(matched_substances) > 1:  # type: ignore
                base_entry = entry[: entry.index(matched_substances[0])]
                return base_entry.strip(), matched_substances
            else:
                idx_match = entry.index(matched_substances[0])
                if idx_match > 0 and entry[idx_match - 2] == ")":
                    return entry.strip(), []
                else:
                    base_entry = entry[: entry.index(matched_substances[0])]
                    return base_entry.strip(), matched_substances
        else:
            return entry.strip(), []

    def get_ending(self, entry: str) -> tuple[str, str]:
        """Extract the ending from an entry."""
        ending_pattern = re.compile(r"\b(" + "|".join(map(re.escape, self.endings)) + r")\b$", re.IGNORECASE)
        match = ending_pattern.search(entry)
        if match:
            ending = match.group(1)
            base_entry = entry[: match.start()].strip()
            return base_entry, ending
        else:
            return entry, ""

    def split_prefix(self, entry: str) -> list[str]:
        """Split the entry based on known prefixes."""
        for prefix in self.prefixes:
            if entry.lower().startswith(prefix + "/") or entry.lower().startswith(prefix + " +"):
                entry_no_prefix = entry[len(prefix) + 1 :].strip()
                entry_splits = entry_no_prefix.split("/")
                if entry_splits[0] in ["va", "vp"]:
                    return [f"{prefix} {entry_splits[0]}"]
                else:
                    return list(filter(len, [f"{prefix} {entry_splits[0]}", "/".join(entry_splits[1:])]))
        return [entry]

    def split_inside_parenthesis(self, ingredients_list: list[str]) -> list[str]:
        """Split ingredients inside parentheses if multiple substances are listed."""
        result_list: list[str] = []

        for word in ingredients_list:
            base_entry, substances = self.get_substance(word.replace("(", "").replace(")", ""))
            if "nano" in word:
                result_list.extend(word)
            elif substances:
                result_list.extend(self.construct_ingredients(base_entry, substances))
            else:
                result_list.extend(self.handle_parenthesis(word))

        return result_list

    def construct_ingredients(self, base_entry: str, substances: list[str]) -> list[str]:
        """Construct ingredients from base entry and substances."""
        ingredients: list[str] = []
        for substance in substances:
            for sub_entry in base_entry.split("/"):
                ingredient = f"{sub_entry.strip()} {substance.strip()}".strip()
                ingredients.append(ingredient)
        return ingredients

    def handle_parenthesis(self, word: str) -> list[str]:
        """Handle splitting inside parenthesis for complex ingredients."""
        result_list: list[str] = []
        matches = re.search(r"(\w+\s+)?\(([^)]+/[^)]+)\)(\s+\w+)?", word)
        if matches:
            match = matches.groups()
            before_word = match[0].strip() if match[0] else ""
            inside_brackets = match[1].split("/")
            after_word = match[2].strip() if match[2] else ""

            for inside_word in inside_brackets:
                inside_word = inside_word.strip()
                if before_word and after_word:
                    result_list.append(f"{before_word} {inside_word} {after_word}")
                elif before_word:
                    result_list.append(f"{before_word} {inside_word}")
                elif after_word:
                    result_list.append(f"{inside_word} {after_word}")
                else:
                    result_list.append(inside_word)
        elif re.search(r"\(([^)]+/[^)]+)\)", word):
            matches = re.findall(r"\(([^)]+/[^)]+)\)", word)[0].split("/")

            base_word = " ".join(matches[0].split(" ")[:-1]).strip()
            first_word = matches[0].split(" ")[-1]
            for i in range(len(matches)):  # type: ignore
                if i == 0:
                    result_list.append(f"{base_word} {first_word.strip()}")
                else:
                    result_list.append(f"{base_word} {matches[i].strip()}")
        else:
            result_list.append(word)

        return result_list

    def split_after_endings(self, entry: str) -> list[str]:
        """Split the entry after each occurrence of the specified endings."""
        pattern = r"\b(" + "|".join(map(re.escape, self.endings)) + r")\b"
        matches = re.findall(pattern, entry, re.IGNORECASE)

        if len(matches) <= 1:  # type: ignore
            return [entry]

        split_entries = re.split(pattern, entry, flags=re.IGNORECASE)

        result_list = []
        for i in range(0, len(split_entries), 2):  # type: ignore
            combined = split_entries[i].strip()
            if i + 1 < len(split_entries):
                combined += " " + split_entries[i + 1].strip()
            result_list.append(combined)

        return result_list

    def get_split(self, input_entry: str) -> list[str]:
        """Process the entry and return the split ingredients."""
        entry = self.clean_entry(input_entry)
        final_ingredients = []

        if "/" in entry:
            pre_ingredients = []
            ingredients, ending = self.get_ending(entry)

            # Find and extract substance ... (e.g. "stem", "leaf", etc.)
            base_ingredients, substance_list = self.get_substance(ingredients)

            # Split strings at "/" outside of parenthesis
            ingredients_list = self.split_outside_parentheses(base_ingredients)

            # Check and split inside parenthesis
            ingredients_list = self.split_inside_parenthesis(ingredients_list)

            # Construct final ingredient
            for ingredient in ingredients_list:
                if substance_list:
                    for substance in substance_list:
                        pre_ingredient = f"{ingredient.strip()} {substance.strip()} {ending.strip()}".strip()
                        pre_ingredients.append(pre_ingredient)
                else:
                    pre_ingredient = f"{ingredient.strip()} {ending.strip()}".strip()
                    pre_ingredients.append(pre_ingredient)

        else:
            pre_ingredients = [entry]

        for ingredient in pre_ingredients:
            final_ingredients.extend(self.split_after_endings(ingredient))

        return final_ingredients

    def get_inci_json(self, simple: bool = True) -> None:
        """Generate inci.json file from .xlsx and optional existing json."""
        inci_entries = self.xlsx_to_list(simple)

        if self.existing_json_path:
            existing_data_set = self.load_existing_json(self.existing_json_path)
            inci_entries = existing_data_set.union(inci_entries)

        inci_list = sorted(set(inci_entries))

        if simple:
            inci_list = sorted([self.clean_entry(entry) for entry in inci_list])

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, mode="w", encoding="utf-8") as json_file:
            json.dump(inci_list, json_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    inci_xlsx_path = default_config.inci_path
    output_path = default_config.inci_path_simple

    process_inci = ProcessInci(inci_xlsx_path, output_path=output_path)
    process_inci.get_inci_json()
