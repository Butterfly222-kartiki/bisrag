import re
from PyPDF2 import PdfReader

reader = PdfReader("dataset.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text() or ""

# Normalize text (VERY IMPORTANT)
text = text.replace("\n", " ")
text = re.sub(r'\s+', ' ', text)

# Regex to capture:
# IS 1489
# IS 1489 (Part 1)
# IS 1489 : 1991
# IS 784 - 2001
pattern = r'IS\s*\d+(?:\s*\(\s*Part\s*\d+\s*\))?(?:\s*[:\-]\s*\d{4})?'

matches = re.findall(pattern, text)

# Clean
cleaned = [m.strip() for m in matches]

# Deduplicate
unique = list(set(cleaned))

print("Total extracted:", len(matches))
print("Unique:", len(unique))

# Save
with open("final_is_list.txt", "w") as f:
    for item in sorted(unique):
        f.write(item + "\n")