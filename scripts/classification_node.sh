python main.py  --type classification_node \
                --data cora \
                --model HyperDrop \
                --num-convs 6 \
                --gpu $1 \
                --experiment-number $2 \
                --lr-schedule \
                --seed 42

# python main.py  --type classification_node \
#                 --data citeseer \
#                 --model HyperDrop \
#                 --num-convs 6 \
#                 --gpu $1 \
#                 --experiment-number $2 \
#                 --lr-schedule \
#                 --seed 42