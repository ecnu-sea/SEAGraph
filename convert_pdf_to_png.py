#!/usr/bin/env python3
"""
Convert PDF files in asset folder to PNG images for README display
Requirements: pip install pdf2image pillow
Also requires poppler-utils: brew install poppler (macOS)
"""

from pdf2image import convert_from_path
import os

# PDF files to convert
pdf_files = [
    'asset/seagraph.pdf',
    'asset/case.pdf', 
    'asset/combined_comparison_heatmaps.pdf',
    'asset/exp3_bar.pdf',
    'asset/exp3_bar_qwen.pdf'
]

for pdf_file in pdf_files:
    if os.path.exists(pdf_file):
        print(f"Converting {pdf_file}...")
        try:
            # Convert PDF to images (high quality)
            images = convert_from_path(pdf_file, dpi=300)
            
            # Save first page as PNG
            output_file = pdf_file.replace('.pdf', '.png')
            images[0].save(output_file, 'PNG', quality=95)
            print(f"✓ Saved to {output_file}")
        except Exception as e:
            print(f"✗ Error converting {pdf_file}: {e}")
    else:
        print(f"✗ File not found: {pdf_file}")

print("\n✨ Conversion complete! Now uncomment the image blocks in README.md")

