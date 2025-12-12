python test.py \
--content_path /mnt/harddisk2/Zhangmengge/codespace/test/test05/content/ \
--style_path /mnt/harddisk2/Zhangmengge/codespace/test/test05/style/ \
--name AdaAttN_test \
--model adaattn \
--dataset_mode unaligned \
--load_size 512 \
--crop_size 512 \
--image_encoder_path /mnt/harddisk2/Zhangmengge/codespace/MyAttn/MyAtt_GSA_contentEnhance_XS/models/vgg_normalised.pth \
--gpu_ids 0