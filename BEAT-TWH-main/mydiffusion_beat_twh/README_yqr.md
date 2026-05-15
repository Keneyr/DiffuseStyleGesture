Adapted version: xx_yqr.py
training with GENEA dataset, the motion file is for Metahuman instead of orginal GENEA motion file,
which means the bone name, the rotation data all changed.

1. test the trained model (the author given) for gt val dataset, after download the requirements files and models:

go to mydiffusion_beat_twh folder

python sample.py --config=./configs/DiffuseStyleGesture.yml --dataset TWH --gpu 0 --model_path './TWH_mymodel4_512_v0/model001200000.pt' --max_len 0 --tst_prefix 'val_2023_v0_014_main-agent'

running successfully, the inference result is saved as bvh (1800 frames), the sequence window is 4s.

2. test the trained model (the author given) for customized audio (could be yourself), after download the requirements files and models:

python transcribe_txt_likegentle_yqr.py tts.wav tts_align_yqr.txt
python transcribe_txt_likegentle_yqr.py tts.wav tts_align_yqr.tsv
python process_text.py

go to mydiffusion_beat_twh folder

python sampl.py --config=./configs/DiffuseStyleGesture.yml --gpu 0 --model_path './TWH_mymodel4_512_v0/model001200000.pt' --max_len 0 --wav_path ../data/tts.wav --txt_path ../data/tts_align_yqr_process.tsv --wavlm_path ../process/WavLM/WavLM-Large.pt --word2vector_path ../process/crawl-300d-2M.vec

3. train the model for the original GENEA dataset:

cd ../process/
python process_TWH_bvh.py --dataroot "/mnt/e/QR/DATASET/Genea2023/" --dataset TWH --save_path /mnt/e/QR/DATASET/Genea2023/processed_DiffuseStyleGesturePlus/" --wavlm_path ../process/WavLM/WavLM-Large.pt --word2vector_path ../process/crawl-300d-2M.vec --gpu 0 --debug False

python calculate_gesture_statistics.py --dataset TWH --version "v0"

the processed dataset is under: E:\QR\DATASET\Genea2023\processed_DiffuseStyleGesture

cd ../mydiffusion_beat_twh
python end2end.py --config=./configs/DiffuseStyleGesture.yml --gpu 0 --dataset TWH

4. train the model for the Metahuman(adapted) GENEA dataset:

only need to process the Metahuman motion file:
cd process/
python process_TWH_bvh_yqr.py --dataroot "/mnt/e/QR/DATASET/Genea2023/" --save_path "/mnt/e/QR/DATASET/Genea2023/processed_DiffuseStyleGesturePlus/" --wavlm_path "./WavLM/WavLM-Large.pt" --word2vector_path "./crawl-300d-2M.vec" --gpu 0

calculate the statistics:
python calculate_gesture_statistics.py --dataset TWH --version "v0_yqr"

then train the model:
cd mydiffusion_beat_twh/
python end2end_yqr.py --config=./configs/DiffuseStyleGesture_yqr.yml --gpu 0 --dataset TWH

use tensforboard watch the loss line:
tensorboard --logdir tb

5. do the inference for the Metahuman(adapted), for gt val dataset:

cd mydiffusion_beat_twh/
python sample_yqr.py --config=./configs/DiffuseStyleGesture_yqr.yml --gpu 0 --model_path './TWH_mymodel4_512_v0_yqr_windows_30_seed_6/model000200000.pt' --max_len 0 --tst_prefix 'val_2023_v0_014_main-agent'

python sample_yqr.py --config=./configs/DiffuseStyleGesture_yqr.yml --gpu 0 --model_path './TWH_mymodel4_512_v0_yqr_windows_60_seed_12/model000200000.pt' --max_len 0 --tst_prefix 'val_2023_v0_014_main-agent'

python sample_yqr.py --config=./configs/DiffuseStyleGesture_yqr.yml --gpu 0 --model_path './TWH_mymodel4_512_v0_yqr_windows_90_seed_18/model000400000.pt' --max_len 0 --tst_prefix 'val_2023_v0_014_main-agent'

python sample_yqr.py --config=./configs/DiffuseStyleGesture_yqr.yml --gpu 0 --model_path './TWH_mymodel4_512_v0_yqr_windows_120_seed_24/model000400000.pt' --max_len 0 --tst_prefix 'val_2023_v0_014_main-agent'

python sample_yqr.py --config=./configs/DiffuseStyleGesture_yqr.yml --gpu 0 --model_path './TWH_mymodel4_512_v0_yqr_windows_150_seed_30/model000400000.pt' --max_len 0 --tst_prefix 'val_2023_v0_014_main-agent'

for batch inferencing the val dataset:

python sample_yqr.py --config=./configs/DiffuseStyleGesture_yqr.yml --gpu 0 --model_path './TWH_mymodel4_512_v0_yqr_windows_30_seed_6/model000200000.pt' --max_len 0 --tst_path '../../TWH_dataset/processed/'

python sample_yqr.py --config=./configs/DiffuseStyleGesture_yqr.yml --gpu 0 --model_path './TWH_mymodel4_512_v0_yqr_windows_150_seed_30/model000400000.pt' --max_len 0 --tst_path '../../TWH_dataset/processed/'

python sample_yqr.py --config=./configs/DiffuseStyleGesture_yqr.yml --gpu 0 --model_path './TWH_mymodel4_512_v0_yqr_windows_120_seed_24/model000400000.pt' --max_len 0 --tst_path '../../TWH_dataset/processed/'