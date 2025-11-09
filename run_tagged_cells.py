import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
import sys
import re
import traceback

def run_tagged_cells(notebook_path, tags=['weekend', 'pipeline_main', 'v85']):
    """Execute only cells with specified tags"""
    print(f"📖 Reading notebook: {notebook_path}")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    print(f"✅ Loaded {len(nb.cells)} total cells")
    print()
    
    # Find tagged cells
    tagged_cells = []
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        
        # Check metadata tags
        cell_tags = cell.get('metadata', {}).get('tags', [])
        
        # Check for comment markers
        has_comment_marker = False
        for tag in tags:
            if f"# TAG: {tag}" in cell.source or f"# @{tag}" in cell.source:
                has_comment_marker = True
                break
        
        # If cell has any of our target tags
        if any(tag in cell_tags for tag in tags) or has_comment_marker:
            tagged_cells.append((idx, cell))
            matching_tags = [t for t in tags if t in cell_tags or f"# TAG: {t}" in cell.source or f"# @{t}" in cell.source]
            print(f"  ✅ Cell {idx}: Found tags {matching_tags}")
            preview = cell.source[:100].replace('\n', ' ')
            print(f"     Preview: {preview}...")
            print()
    
    if not tagged_cells:
        print("❌ ERROR: No tagged cells found!")
        print()
        print(f"Looking for cells with tags: {tags}")
        print()
        print("Available tags in notebook:")
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                cell_tags = cell.get('metadata', {}).get('tags', [])
                if cell_tags:
                    print(f"  Cell {idx}: {cell_tags}")
                # Check for comment markers
                if '# TAG:' in cell.source or '# @' in cell.source:
                    markers = re.findall(r'# TAG: (\w+)|# @(\w+)', cell.source)
                    if markers:
                        print(f"  Cell {idx}: {[m for m in markers[0] if m]} (comment marker)")
        print()
        print("💡 TIP: Add tags to cells in Jupyter:")
        print("   1. Select cell")
        print("   2. View → Cell Toolbar → Tags")
        print("   3. Add tag: 'weekend' or 'pipeline_main'")
        print()
        print("   OR add comment marker at top of cell:")
        print("   # TAG: weekend")
        return False
    
    print(f"🎯 Found {len(tagged_cells)} tagged cells to execute")
    print("=" * 60)
    print()
    
    # Execute tagged cells
    ep = ExecutePreprocessor(
        timeout=1200,  # 20 minutes per cell
        kernel_name='python3',
        allow_errors=False
    )
    
    success_count = 0
    for idx, (cell_idx, cell) in enumerate(tagged_cells):
        print(f"▶️  Executing cell {idx + 1}/{len(tagged_cells)} (original cell {cell_idx})...")
        
        try:
            # Create a temporary notebook with just this cell
            temp_nb = nbformat.v4.new_notebook()
            temp_nb.cells = [cell]
            
            # Execute it
            ep.preprocess(temp_nb, {'metadata': {'path': '.'}})
            
            print(f"   ✅ Cell {cell_idx} executed successfully")
            print()
            success_count += 1
            
        except CellExecutionError as e:
            print(f"   ❌ ERROR in cell {cell_idx}!")
            print(f"   Error type: CellExecutionError")
            print(f"   Error message: {str(e)}")
            print()
            print("   Full traceback:")
            print(traceback.format_exc())
            print()
            print("   Cell source (first 500 chars):")
            print(cell.source[:500])
            print()
            return False
        
        except Exception as e:
            print(f"   ❌ UNEXPECTED ERROR in cell {cell_idx}!")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            print()
            print("   Full traceback:")
            print(traceback.format_exc())
            print()
            print("   Cell source (first 500 chars):")
            print(cell.source[:500])
            print()
            return False
    
    print("=" * 60)
    print(f"✅ Successfully executed {success_count}/{len(tagged_cells)} tagged cells")
    return True

if __name__ == "__main__":
    notebook = sys.argv[1] if len(sys.argv) > 1 else "AI_Forex_Brain_2.ipynb"
    
    # Tags to look for (in order of priority)
    tags = ['weekend', 'pipeline_main', 'v85', 'saturday', 'sunday']
    
    success = run_tagged_cells(notebook, tags)
    sys.exit(0 if success else 1)
