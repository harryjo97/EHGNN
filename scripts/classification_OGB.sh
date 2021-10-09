# # HIV
# python main.py  --type classification_OGB \
#                 --data ogbg-molhiv \
#                 --model HyperDrop \
#                 --gpu $1 \
#                 --experiment-number $2 \
#                 --batch-size 512 \
#                 --num-hidden 128 \
#                 --lr-schedule \
#                 --edge-ratio 0.95 \
#                 --seed 42

# Tox21
python main.py  --type classification_OGB \
                --data ogbg-moltox21 \
                --model HyperDrop \
                --gpu $1 \
                --experiment-number $2 \
                --batch-size 128 \
                --num-hidden 128 \
                --lr-schedule \
                --edge-ratio 0.45 \
                --seed 42 

# # Toxcast
# python main.py  --type classification_OGB \
#                 --data ogbg-moltoxcast \
#                 --model HyperDrop \
#                 --gpu $1 \
#                 --experiment-number $2 \
#                 --batch-size 128 \
#                 --num-hidden 128 \
#                 --lr-schedule \
#                 --edge-ratio 0.85 \
#                 --seed 42

# # BBBP
# python main.py  --type classification_OGB \
#                 --data ogbg-molbbbp \
#                 --model HyperDrop \
#                 --gpu $1 \
#                 --experiment-number $2 \
#                 --batch-size 128 \
#                 --num-hidden 128 \
#                 --lr-schedule \
#                 --edge-ratio 0.65 \
#                 --seed 42