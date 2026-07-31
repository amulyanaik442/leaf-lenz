import json, os
base = os.environ.get("TEMP", os.path.expanduser("~\\AppData\\Local\\Temp")) + "\\opencode\\leaf-lenz"
with open(os.path.join(base, "detector", "ml_assets", "wheat", "wheat_class_names.json")) as f:
    classes = json.load(f)
with open(os.path.join(base, "detector", "disease_data.json")) as f:
    disease_data = json.load(f)

print("Wheat class names:", classes)
print()
for cls in classes:
    label = "Wheat___" + cls.replace(" ", "_")
    found = label in disease_data
    status = "OK" if found else "MISSING"
    print(f"  {label:40s} [{status}]")
