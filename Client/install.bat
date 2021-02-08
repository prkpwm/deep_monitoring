@echo off
CALL  %userprofile%\anaconda3\Scripts\activate.bat %userprofile%\anaconda3\
CALL  conda create --name deep_monitoring python==3.7.7
CALL  conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
CALL  pip install opencv-python==4.4.0.46
CALL  pip install imutils
CALL  pip install tqdm
CALL  pip install PyYAML
CALL  conda install -c conda-forge dlib  
CALL  pip install matplotlib
CALL  pip install scipy
CALL  pip install panda
@pause



