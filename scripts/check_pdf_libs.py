import importlib.util

for name in ["pypdf", "PyPDF2", "fitz", "pdfplumber"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}: {bool(spec)}")
