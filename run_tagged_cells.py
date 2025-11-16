import nbformat
import sys
import os
from datetime import datetime

def extract_and_run_tagged_cells(notebook_path):
    """Extract and execute cells with TAG: pipeline_main"""
    print(f"📖 Reading notebook: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    tagged_cells = []
    cell_numbers = []
    
    # Find cells with TAG: pipeline_main
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            source = cell.source
            if '# TAG: pipeline_main' in source or '#TAG: pipeline_main' in source:
                tagged_cells.append(source)
                cell_numbers.append(i + 1)
    
    if not tagged_cells:
        print("❌ No tagged cells found with 'TAG: pipeline_main'")
        return False
    
    print(f"✅ Found {len(tagged_cells)} tagged cell(s): {cell_numbers}")
    print("=" * 70)
    print()
    
    # Execute each tagged cell
    for idx, (cell_num, code) in enumerate(zip(cell_numbers, tagged_cells), 1):
        print(f"🔄 Executing tagged cell {idx}/{len(tagged_cells)} (cell #{cell_num})")
        print("-" * 70)
        
        # Remove the TAG comment before execution
        code_lines = [line for line in code.split('\n') 
                     if not line.strip().startswith('# TAG:')]
        clean_code = '\n'.join(code_lines)
        
        try:
            # Execute the code
            exec(clean_code, globals())
            print("-" * 70)
            print(f"✅ Cell {idx} completed successfully")
            print("=" * 70)
            print()
        except Exception as e:
            print("-" * 70)
            print(f"❌ Cell {idx} failed with error:")
            print(f"   {type(e).__name__}: {str(e)}")
            print("=" * 70)
            return False
    
    print()
    print("=" * 70)
    print("✅ ALL TAGGED CELLS EXECUTED SUCCESSFULLY")
    print("=" * 70)
    return True

if __name__ == "__main__":
    notebook = sys.argv[1] if len(sys.argv) > 1 else "AI_Forex_Brain_2.ipynb"
    
    if not os.path.exists(notebook):
        print(f"❌ ERROR: Notebook not found: {notebook}")
        sys.exit(1)
    
    print("=" * 70)
    print("🎯 WEEKEND MODE - TAGGED CELLS EXECUTION")
    print("=" * 70)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"📓 Notebook: {notebook}")
    print(f"🎯 Target: Cells with 'TAG: pipeline_main'")
    print("=" * 70)
    print()
    
    success = extract_and_run_tagged_cells(notebook)
    
    print()
    print("=" * 70)
    if success:
        print("✅ WEEKEND EXECUTION COMPLETED")
    else:
        print("❌ WEEKEND EXECUTION FAILED")
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)
    
    sys.exit(0 if success else 1)
