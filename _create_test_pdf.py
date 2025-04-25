import sys
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

def create_pdf(filename="test_corpus.pdf"):
    """Creates a simple PDF file with some text."""
    try:
        c = canvas.Canvas(filename, pagesize=letter)
        textobject = c.beginText()
        textobject.setTextOrigin(inch, 10*inch)
        textobject.setFont("Helvetica", 12)

        lines = [
            "PDF Corpus Content",
            "",
            "This is the first line of text in the PDF.",
            "This is the second line.",
            "And a third line for good measure.",
        ]

        for line in lines:
            textobject.textLine(line)

        c.drawText(textobject)
        c.save()
        print(f"Successfully created PDF: {filename}")
        return 0
    except Exception as e:
        print(f"Error creating PDF {filename}: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    # Add reportlab if missing
    try:
        import reportlab
    except ImportError:
        print("ReportLab not found, attempting to install...", file=sys.stderr)
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "install", "reportlab"], capture_output=True, text=True)
        if result.returncode != 0:
             print(f"Failed to install reportlab: {result.stderr}", file=sys.stderr)
             sys.exit(1)
        print("ReportLab installed successfully.", file=sys.stderr)

    sys.exit(create_pdf())