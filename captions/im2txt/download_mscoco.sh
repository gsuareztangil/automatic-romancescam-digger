cd im2txt-workspace

# Location to save the MSCOCO data.
MSCOCO_DIR="${HOME}/mscoco"

# Build the preprocessing script.
bazel build im2txt/download_and_preprocess_mscoco

# Run the preprocessing script.
bazel-bin/im2txt/download_and_preprocess_mscoco "${MSCOCO_DIR}"

cd ..