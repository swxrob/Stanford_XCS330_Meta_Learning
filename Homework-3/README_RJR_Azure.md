
0) Azure Guide
https://docs.google.com/document/d/11jTRDhSuuhw3PYyJYl8_XFVqIsUiur5cjATMVKNNfI4/edit

1) Open Azure Control Panel
https://labs.azure.com/virtualmachines

2) SCP recursively:
scp -r -P 60039 Homework-3.zip scpdxcs@ml-lab-f1d25ece-3557-428f-8c40-37102309de76.eastus.cloudapp.azure.com:XCS330/Homework-3/

3) SSH Tunnel for Tensorboard:
# On laptop:
ssh -p 60039 -N -f -L localhost:6007:localhost:6007 scpdxcs@ml-lab-f1d25ece-3557-428f-8c40-37102309de76.eastus.cloudapp.azure.com

4) SSH into VM:
ssh -p 60039 scpdxcs@ml-lab-f1d25ece-3557-428f-8c40-37102309de76.eastus.cloudapp.azure.com
sp@c3WXp@p3r

5) Tmux:
tmux
ctrl-b-%

6) Activate Conda
conda activate XCS330_CUDA

7) Start Tensorboard:
# On Azure VM:
tensorboard --logdir logs/ --port 6007

8) Run Scripts:
./run_all.sh

9) Open Browser Tensorboard on Laptop

