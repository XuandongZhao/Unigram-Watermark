gdrive_download () {
  CONFIRM=$(curl -sc /tmp/gdrive_cookies "https://drive.google.com/uc?export=download&id=$1" | grep -o 'confirm=[^&]*' | cut -d '=' -f 2)
  curl -Lb /tmp/gdrive_cookies "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -o $2
}

mkdir LFQA
mkdir OpenGen

# download data from https://drive.google.com/drive/folders/1mPROenBB0fzLO9AX4fe71k0UYv0xt3X1
# using lfqa-data/inputs.jsonl and opengen-data/inputs.jsonl
gdrive_download 1wYlYnIbfWBsJxiL1DH12H92kVg0swCDQ ./LFQA/inputs.jsonl
gdrive_download 1TSoZowrCmniOyoaGXeHG62KOktFXbm9P ./OpenGen/inputs.jsonl