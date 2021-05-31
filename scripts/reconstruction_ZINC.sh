# ZINC
python main.py  --type reconstruction_ZINC \
                --data ZINC \
                --model HyperCluster \
                --gpu $1 \
                --experiment-number $2 \
                --batch-size 128 \
                --num-hidden 32 \
                --lr 0.001 \
                --edge-ratio 0.15 \
                --num-heads 1 \
                --seed 42 \
                --cluster \
                --lr-schedule