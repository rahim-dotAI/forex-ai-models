import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
import sys
import os
from pathlib import Path

def run_notebook(notebook_path):
    """Execute entire Jupyter notebook"""
    print(f"📖 Reading notebook: {notebook_path}")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    print(f"✅ Loaded {len(nb.cells)} cells")
    print(f"   - Code cells: {sum(1 for c in nb.cells if c.cell_type == 'code')}")
    print(f"   - Markdown cells: {sum(1 for c in nb.cells if c.cell_type == 'markdown')}")
    print()
    
    # Configure executor
    ep = ExecutePreprocessor(
        timeout=2400,  # 40 minutes per cell
        kernel_name='python3',
        allow_errors=False  # Stop on first error
    )
    
    print("🚀 Executing notebook...")
    print("=" * 60)
    
    try:
        # Execute the notebook
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        
        print()
        print("=" * 60)
        print("✅ Notebook execution completed successfully!")
        
        # Save executed notebook
        output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"💾 Saved executed notebook: {output_path}")
        
        return True
        
    except CellExecutionError as e:
        print()
        print("=" * 60)
        print(f"❌ ERROR: Cell execution failed!")
        print(f"Cell index: {e.cell_index if hasattr(e, 'cell_index') else 'unknown'}")
        print(f"Error: {str(e)}")
        print("=" * 60)
        return False
    
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ ERROR: Unexpected error during execution!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print("=" * 60)
        return False

if __name__ == "__main__":
    notebook = sys.argv[1] if len(sys.argv) > 1 else "AI_Forex_Brain_2.ipynb"
    
    if not os.path.exists(notebook):
        print(f"❌ ERROR: Notebook not found: {notebook}")
        sys.exit(1)
    
    success = run_notebook(notebook)
    sys.exit(0 if success else 1)
