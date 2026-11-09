"""
setup_dataset.py - Quick setup script to prepare the recipe dataset.

Extracts recipes_raw.zip (if recipes_raw/ is not extracted yet)
and builds data/processed/recipes.pkl (122,938 clean recipes).
"""
import zipfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "recipes_raw"
ZIP_PATH = ROOT / "recipes_raw.zip"
PROCESSED_FILE = ROOT / "data" / "processed" / "recipes.pkl"


def main():
    print("=" * 60)
    print("🍳 Nosh Recipe Assistant — Dataset Initialization")
    print("=" * 60)

    # 1. Check if processed cache already exists
    if PROCESSED_FILE.exists():
        print(f"✅ Clean dataset cache already exists at {PROCESSED_FILE.relative_to(ROOT)}")
        print("All 122,938 recipes are ready for instant startup!")
        return

    # 2. Check if raw JSON directory exists
    if not (RAW_DIR / "recipes_raw_nosource_ar.json").exists():
        if ZIP_PATH.exists():
            print(f"📦 Extracting {ZIP_PATH.name} into recipes_raw/ ...")
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(ZIP_PATH, "r") as z:
                z.extractall(RAW_DIR)
            print("✅ Extraction complete.")
        else:
            print(f"❌ Error: Neither {RAW_DIR} nor {ZIP_PATH} found!")
            print("Please place recipes_raw.zip or the recipes_raw/ folder in the project root.")
            sys.exit(1)

    # 3. Process raw recipes and build recipes.pkl cache
    print("⚡ Parsing and cleaning 125,000+ raw records into validated dataset...")
    sys.path.insert(0, str(ROOT))
    from app.recipe_loader import get_recipes
    recipes = get_recipes(raw_dir=RAW_DIR, processed_path=PROCESSED_FILE, force_reload=True)
    print(f"🎉 Success! Generated {len(recipes):,} validated recipes in {PROCESSED_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
