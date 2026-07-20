import os
import nbformat
from nbmerge import merge_notebooks

# Find all .ipynb files in the current folder, ignoring any previous merged files
notebook_files = [f for f in os.listdir('.') if f.endswith('.ipynb') and f != 'merged_all_notebooks.ipynb']
notebook_files.sort()  # Sorts them alphabetically/numerically

print(f"Found {len(notebook_files)} notebooks to merge...")

if not notebook_files:
    print("❌ No .ipynb files found in this folder!")
else:
    # Merge the notebooks using nbmerge's internal function
    merged_nb = merge_notebooks('.', notebook_files)
    
    # Save the combined notebook structure
    with open('merged_all_notebooks.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(merged_nb, f)
        
    print("🎉 Successfully created: merged_all_notebooks.ipynb")
