for expected_batchsize in 100
do

for epsilon in 5
do

for EPOCH in 50
do

for lr in 1
do 

for aug_num in 35
do

export CUDA_VISIBLE_DEVICES = 0
python main.py --expected_batchsize $expected_batchsize --epsilon $epsilon --EPOCH $EPOCH --lr $lr --aug_num $aug_num --log_dir logs 


done
done
done
done
done


# srun --time=08:00:00 --reservation=A100 --gres=gpu:a100:1  --mem=50G --resv-ports=1 --pty /bin/bash -l
# jupyter notebook --no-browser --ip=0.0.0.0
# ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9