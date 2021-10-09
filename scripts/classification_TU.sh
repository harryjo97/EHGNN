# # D&D
# python main.py  --type classification_TU \
#                 --data DD \
#                 --model HyperDrop \
#                 --gpu $1 \
#                 --experiment-number $2 \
#                 --batch-size 10 \
#                 --num-hidden 32 \
#                 --lr-schedule \
#                 --edge-ratio 0.75 \
#                 --seed 42

# PROTEINS
python main.py  --type classification_TU \
                --data PROTEINS \
                --model HyperDrop \
                --gpu $1 \
                --experiment-number $2 \
                --batch-size 128 \
                --num-hidden 128 \
                --lr-schedule \
                --edge-ratio 0.65 \
                --seed 42

# # MUTAG
# python main.py  --type classification_TU \
#                 --data MUTAG \
#                 --model HyperDrop \
#                 --gpu $1 \
#                 --experiment-number $2 \
#                 --batch-size 128 \
#                 --num-hidden 128 \
#                 --lr-schedule \
#                 --edge-ratio 0.9 \
#                 --seed 42

# # IMDB-B
# python main.py  --type classification_TU \
#                 --data IMDB-BINARY \
#                 --model HyperDrop \
#                 --gpu $1 \
#                 --experiment-number $2 \
#                 --batch-size 128 \
#                 --num-hidden 128 \
#                 --lr-schedule \
#                 --edge-ratio 0.9 \
#                 --seed 42

# # IMDB-M
# python main.py  --type classification_TU \
#                 --data IMDB-MULTI \
#                 --model HyperDrop \
#                 --gpu $1 \
#                 --experiment-number $2 \
#                 --batch-size 128 \
#                 --num-hidden 128 \
#                 --lr-schedule \
#                 --edge-ratio 0.85 \
#                 --seed 42

# # COLLAB
# python main.py  --type classification_TU \
#                 --data COLLAB \
#                 --model HyperDrop \
#                 --gpu $1 \
#                 --experiment-number $2 \
#                 --batch-size 128 \
#                 --num-hidden 128 \
#                 --lr-schedule \
#                 --edge-ratio 0.85 \
#                 --seed 42