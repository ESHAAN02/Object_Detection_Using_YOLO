from tqdm import tqdm #pip install tqdm
import requests #pip install requests
chunk_size=1024
url="https://pjreddie.com/media/files/yolov3.weights"
r=requests.get(url,stream=True)
total_size=int(r.headers['content-length'])
with open("python_download","wb") as f:
    for data in tqdm(iterable=r.iter_content(chunk_size=chunk_size),total=total_size/chunk_size,unit="KB"):
        f.write(data)
print("download complete")

