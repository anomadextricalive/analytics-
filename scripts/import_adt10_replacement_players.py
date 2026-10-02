"""Export the public scouting fields from the ADT10 replacement workbook."""

import csv
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path.home() / "Downloads" / "2026 ADT10 Replacement Player List 1.10.26.xlsx"
DEFAULT_OUTPUT = ROOT / "data" / "adt10_replacement_players.csv"
NAMESPACE = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
EXPORT_COLUMNS = [
    "First Name", "Last Name", "Age", "Cricket Board", "Draft Category",
    "Player Role", "Batting Style", "Bowling Style", "T20 Internationals",
    "T20 Domestic Matches", "Cricinfo Profile Link", "Availability",
    "Availability Notes", "Nationality",
]


def column_index(cell_ref):
    letters = re.match(r"[A-Z]+", cell_ref).group()
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def cell_value(cell, shared_strings):
    kind = cell.attrib.get("t")
    if kind == "inlineStr":
        return "".join(part.text or "" for part in cell.findall(".//m:t", NAMESPACE))
    value = cell.find("m:v", NAMESPACE)
    if value is None or value.text is None:
        return ""
    if kind == "s":
        return shared_strings[int(value.text)]
    return value.text


def read_all_players(source):
    with zipfile.ZipFile(source) as workbook:
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            strings = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(part.text or "" for part in item.findall(".//m:t", NAMESPACE))
                for item in strings.findall("m:si", NAMESPACE)
            ]

        workbook_xml = ET.fromstring(workbook.read("xl/workbook.xml"))
        sheets = workbook_xml.find("m:sheets", NAMESPACE)
        sheet_id = next(
            sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            for sheet in sheets
            if sheet.attrib.get("name") == "All Players"
        )
        relationships = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        target = next(
            rel.attrib["Target"]
            for rel in relationships
            if rel.attrib.get("Id") == sheet_id
        ).lstrip("/")
        sheet_path = target if target.startswith("xl/") else f"xl/{target}"
        sheet = ET.fromstring(workbook.read(sheet_path))
        rows = sheet.findall(".//m:sheetData/m:row", NAMESPACE)
        header_cells = rows[0].findall("m:c", NAMESPACE)
        headers = {
            column_index(cell.attrib["r"]): cell_value(cell, shared_strings)
            for cell in header_cells
        }

        records = []
        for row in rows[1:]:
            cells = row.findall("m:c", NAMESPACE)
            values = {
                headers[column_index(cell.attrib["r"])]: cell_value(cell, shared_strings)
                for cell in cells
                if column_index(cell.attrib["r"]) in headers
            }
            record = {column: values.get(column, "") for column in EXPORT_COLUMNS}
            has_scouting_data = any(record[column] for column in EXPORT_COLUMNS[2:])
            if (record["First Name"] or record["Last Name"]) and has_scouting_data:
                records.append(record)
        return records


def main():
    source = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_SOURCE
    output = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else DEFAULT_OUTPUT
    records = read_all_players(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Exported {len(records)} players to {output}")


if __name__ == "__main__":
    main()
