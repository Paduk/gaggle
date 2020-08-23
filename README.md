# change py to 3.5
sudo update-alternatives --config python <br />

# in case of pip error
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py <br />
python3 get-pip.py --force-reinstall <br />
sudo apt-get install python3-lxml <br />
pip3 install lxml <br />
pip3 install nltk <br />
pip3 install gensim <br />
pip3 install konlpy <br />
pip3 install matplotlib pandas <br />

# in case of jupyter error after changing python3 version
pip3 install --upgrade notebook <br />
pip3 install lesscpy <br />

# in case of nltk download error
import nltk <br />
nltk.download('punkt') <br />

# need to understand below function
<h4> from nltk.tokenize import word_tokenize, sent_tokenize </h4>
<h4> 한국어형태소 function usage </h4>
<h4></h4>
