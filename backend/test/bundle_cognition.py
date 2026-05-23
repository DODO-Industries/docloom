import os

def bundle_cognition():
    root = r"d:\persnol\DocLoom"
    cognition_dir = os.path.join(root, "backend", "services", "loom_service", "cognition")
    test_file = os.path.join(root, "backend", "services", "loom_service", "test_cognition_arch.py")
    output_file = os.path.join(root, "assets", "cognition_codebase.txt")
    
    files_to_bundle = []
    if os.path.exists(cognition_dir):
        for f in os.listdir(cognition_dir):
            if f.endswith(".py"):
                files_to_bundle.append(os.path.join(cognition_dir, f))
    
    if os.path.exists(test_file):
        files_to_bundle.append(test_file)
        
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("==================================================\n")
        out.write("DOCLOOM COGNITIVE ARCHITECTURE BUNDLE\n")
        out.write("==================================================\n\n")
        
        for file_path in files_to_bundle:
            file_name = os.path.basename(file_path)
            relative_path = os.path.relpath(file_path, root)
            
            out.write(f"\n{'#'*80}\n")
            out.write(f"# FILE: {file_name}\n")
            out.write(f"# PATH: {relative_path}\n")
            out.write(f"{'#'*80}\n\n")
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    out.write(f.read())
            except Exception as e:
                out.write(f"ERROR READING FILE: {e}")
            
            out.write("\n\n")

if __name__ == "__main__":
    bundle_cognition()
    print("Bundle created successfully at assets/cognition_codebase.txt")
