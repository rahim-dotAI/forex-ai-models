import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys, os, time, re
from datetime import datetime

class TaggedCellExecutor(ExecutePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tagged_cells = []
        self.current_cell = 0
        self.start_time = None
        
    def preprocess(self, nb, resources=None, km=None):
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == 'code' and '#TAG: pipeline_main' in cell.source:
                self.tagged_cells.append(idx)
        
        print(f"📊 Found {len(self.tagged_cells)} tagged cells")
        self.start_time = time.time()
        return super().preprocess(nb, resources, km)
    
    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type != 'code' or cell_index not in self.tagged_cells:
            return cell, resources
        
        self.current_cell += 1
        print(f"\n🔄 Tagged Cell {self.current_cell}/{len(self.tagged_cells)}")
        
        cell_start = time.time()
        cell, resources = super().preprocess_cell(cell, resources, cell_index)
        cell_time = time.time() - cell_start
        
        if cell.outputs:
            for output in cell.outputs:
                if output.output_type == 'stream':
                    print(re.sub(r'\x1b\[[0-9;]*m', '', output.text))
                elif output.output_type == 'error':
                    print(f"❌ {output.ename}: {output.evalue}")
        
        print(f"✅ Completed in {cell_time:.1f}s")
        return cell, resources

def run_tagged_cells(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = TaggedCellExecutor(timeout=2400, kernel_name='python3', allow_errors=False)
    
    print("🚀 Starting tagged cells execution...")
    start = time.time()
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    duration = time.time() - start
    
    print(f"\n✅ COMPLETED in {duration:.1f}s")
    return True

if __name__ == "__main__":
    notebook = "AI_Forex_Brain_2.ipynb"
    if not os.path.exists(notebook):
        print(f"❌ Not found: {notebook}")
        sys.exit(1)
    
    success = run_tagged_cells(notebook)
    sys.exit(0 if success else 1)
