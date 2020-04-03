cd ~/Desktop/3d_recognition_thesis/2d_recognition_thesis
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt | grep -v 'already satisfied'
python3 2d_recog.py