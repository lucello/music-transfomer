import kagglehub

# Download latest version
path = kagglehub.dataset_download("soumikrakshit/classical-music-midi")

print("Path to dataset files:", path)
