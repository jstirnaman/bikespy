1. Create a directory and a conda virtual environment.
   ```sh
   mkdir bikespy && cd bikespy
   conda create -p envs/pytorch-env
   conda activate ./envs/pytorch-env
   
1. Use conda to install pytorch and Torchvision.
   ```sh
   conda install pytorch torchvision -c pytorch
   ```

1. Read the Torchvision object detection tutorial and download the training dataset.
   ```sh
   cd ./img-training && curl -l https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip && unzip PennFudanPed.zip && cd ../
   ```

1. Download bike pics from the archives
   ```sh
   cd scrapers && npm install
   ```
   ```sh
   node index.js
   ```
