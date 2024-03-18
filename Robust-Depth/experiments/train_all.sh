python -u train.py \
--model_name tes \
--height 192 \
--width 640 \
--cuda 0 \
--log_frequency 3500 \
--num_epochs 30 \
--use_augpose_loss \
--do_fog --do_snow --do_rain --do_flip

