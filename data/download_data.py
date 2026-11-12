import kagglehub

# Download dataset
path = kagglehub.dataset_download(
    "matthewwilliams333/billboard-hot-100-year-end-tracks-2000-2024",
    output_dir="./billboard"
)

print("Path to dataset files:", path)