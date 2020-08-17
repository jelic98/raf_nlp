import gdown
url = 'https://drive.google.com/uc?id=1fGYzEkGSTLBDFSRRiQw9jeaC9xHWws2g'
output = 'res.zip'
gdown.download(url, output, quiet=False)
