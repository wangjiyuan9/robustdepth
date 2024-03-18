python -u train.py \
--model_name tes \
--height 192 \
--width 640 \
--cuda 0 \
--log_frequency 3500 \
--num_epochs 30 \
--use_augpose_loss \
--do_gauss --do_shot --do_impulse --do_defocus --do_glass \
--do_zoom --do_snow --do_frost --do_elastic --do_pixelate \
--do_jpeg_comp --do_color --do_blur --do_night --do_rain \
--do_scale --do_tiling --do_vertical --do_erase --do_flip \
--do_greyscale --do_ground_snow --do_dusk --do_dawn --do_fog \
--R --G --B

