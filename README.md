sudo apt-get install language-pack-ko &&
sudo locale-gen ko_KR.UTF-8 &&
sudo apt-get update &&
sudo apt-get install python3-pip -y &&
sudo add-apt-repository ppa:deadsnakes/ppa &&
sudo apt-get update &&
sudo apt-get install python3.6

sudo update-alternatives --install /usr/bin/python3 python /usr/bin/python3.5 2   
sudo update-alternatives --install /usr/bin/python3 python /usr/bin/python3.6 1
sudo update-alternatives --config python
# python3.6 install fin
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.1.243-1_amd64.deb
sudo dpkg -i ./cuda-repo-ubuntu1604_10.1.243-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-10-1
sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" >> /etc/apt/sources.list.d/cuda.list'
nvidia-smi
sudo apt-get update
sudo apt install libcudnn7=7.6.2.24-1+cuda10.1
# gpu install fin

pip3 install --upgrade pip
pip3 install tensorflow-gpu torch torchvision keras
sudo apt install python3.6-gdbm

pip3 install jupyter sklearn matplotlib seaborn pandas
pip3 install librosa
# package install fin

jupyter notebook --generate-config
vi ~/.jupyter/jupyter_notebook_config.py
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
c = get_config()
c.NotebookApp.ip = '34.68.167.255'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888

# jupyter setup fin
pip install --user kaggle


2. kaggle홈페이지 -> myaccount -> get api token -> kaggle.json 파일을 gcp내 .kaggle/폴더에 복사
3. 명령어 실행 (ex: kaggle competitions download -c tensorflow-speech-recognition-challenge) 

gpu check
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
K.tensorflow_backend._get_available_gpus()