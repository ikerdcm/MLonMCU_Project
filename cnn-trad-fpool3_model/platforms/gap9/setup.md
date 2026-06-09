# MLonMCU FS2026 - GAP9 


## Install Docker
https://docs.docker.com/engine/install/

## Setup Docker

sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world

## Setup ssh key for access to the private GAP_SDK

echo "-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACDAzzz54yJ2g06bNQdIlr8gXd5lKuK/jR1W4fB3dNOYbwAAAJjCKDtOwig7
TgAAAAtzc2gtZWQyNTUxOQAAACDAzzz54yJ2g06bNQdIlr8gXd5lKuK/jR1W4fB3dNOYbw
AAAEDgke1DZRXjntjZSlF2cdTSrfInP41bCPPUhze3jFKkv8DPPPnjInaDTps1B0iWvyBd
3mUq4r+NHVbh8Hd005hvAAAAFHdpZXNlcEBNYWMuZnJpdHouYm94AQ==
-----END OPENSSH PRIVATE KEY-----" >> ~/.ssh/mlonmcu_gap-sdk
chmod 600 ~/.ssh/mlonmcu_gap-sdk
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMDPPPnjInaDTps1B0iWvyBd3mUq4r+NHVbh8Hd005hv mlonmcu" >> ~/.ssh/mlonmcu_gap-sdk.pub

## Clone repo and make container
git clone https://github.com/pulp-platform/Deeploy.git
cd Deeploy/Container
make deeploy-gap9 SSH_PRIVATE_KEY="~/.ssh/mlonmcu_gap-sdk"


## Run container
docker run -it --name deeploy_gap9 -v $(pwd):/app/Deeploy ghcr.io/pulp-platform/deeploy-gap9:latest
### Although not necessary, you might find it useful to run the container in privileged mode
docker run -it --privileged --name deeploy_gap9 -v $(pwd):/app/Deeploy ghcr.io/pulp-platform/deeploy-gap9:late


## Before running tests, you need to set up the GAP9 environment inside the container:
source /app/install/gap9-sdk/.gap9-venv/bin/activate
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation


## Test helloworld in GVSoC
cd /app/install/gap9-sdk/examples/gap9/basic/helloworld/
cmake -B build
cmake --build build --target menuconfig # GAP_SDK > Platform > Platform > GVSoC/Board > Esc
cmake --build build --target run




## Test helloworld on board

### If you are using WSL, first attach the USB device
https://learn.microsoft.com/en-us/windows/wsl/connect-usb

## Open two more terminals in your Deeploy directory

### Make sure that libusb is installed
apt-get install libusb-1.0.-0-dev # Linux
brew install libusb # macOS

### In the second terminal run
./scripts/gap9-run.sh start-usbip-host
### This will open a connection on 127.0.0.1
### Other addresses I tested are 0.0.0.0 and 172.17.0.1
### To change the address, edit `.pyusbip/pyusbip.py` as follows:
USBIP_HOST='your_address' # replace with your desired address


## In the third terminal
### If you are using a Linux machine, first run
sudo modprobe vhci-hcd

### Run
./scripts/gap9-run.sh --host your_address start

### Alternative for manual setup
#### In case you see "usbip: error: could not connect to `your_address`: System error"
#### run `lsusb` to make sure that the device is visible. You should see
Bus XXX, Device YYY: ID 0403:6011 Future Technology Devices International...
#### Run, replacing accordingly
usbip attach --remote=your_address --busid=XXX-YYY
#### Comment out `cmd_attach_usbip` on line 468 in `./scripts/gap9-run.sh`
#### Open a fourth terminal and run
./scripts/gap9-run.sh --host your_address start


#### In the container, repeat the helloworld steps, selecting `board` as deployment platform
