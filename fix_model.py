import zipfile
import json
from pathlib import Path

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_MODEL = BASE_DIR / "biskraPalmNet.keras"
OUTPUT_MODEL = BASE_DIR / "biskraPalmNet_fixed.keras"


# ============================================================
# Check input model
# ============================================================

if not INPUT_MODEL.exists():

    print("ERROR: biskraPalmNet.keras was not found.")

    print(
        f"Expected location:\n{INPUT_MODEL}"
    )

    raise SystemExit(1)


print("Original model:")
print(INPUT_MODEL)

print("\nCreating compatible model...")


# ============================================================
# Read .keras archive
# ============================================================

with zipfile.ZipFile(
    INPUT_MODEL,
    "r"
) as zin:

    # Read model configuration
    config_data = zin.read(
        "config.json"
    )

    config = json.loads(
        config_data.decode("utf-8")
    )

    # --------------------------------------------------------
    # Remove incompatible quantization_config entries
    # --------------------------------------------------------

    removed = 0

    def clean_config(obj):

        nonlocal_dummy = None

        global removed

        if isinstance(obj, dict):

            if "quantization_config" in obj:

                del obj[
                    "quantization_config"
                ]

                removed += 1

            for value in obj.values():

                clean_config(value)

        elif isinstance(obj, list):

            for item in obj:

                clean_config(item)


    clean_config(config)

    print(
        f"Removed incompatible entries: {removed}"
    )

    # --------------------------------------------------------
    # Create fixed .keras file
    # --------------------------------------------------------

    with zipfile.ZipFile(
        OUTPUT_MODEL,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as zout:

        for item in zin.infolist():

            data = zin.read(
                item.filename
            )

            # Replace original config
            if item.filename == "config.json":

                data = json.dumps(
                    config,
                    ensure_ascii=False,
                    separators=(",", ":")
                ).encode("utf-8")

            zout.writestr(
                item,
                data
            )


print("\nSUCCESS!")
print(
    f"Fixed model created at:\n{OUTPUT_MODEL}"
)