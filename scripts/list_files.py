import os

def list_files(root):
    print(f"\n📂 Listing files under: {root}\n")
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        for f in filenames:
            print(os.path.join(rel, f))

if __name__ == "__main__":
    list_files("data_prep")
