export DIR=
export MODEL_NAME=glyce
export FONT_TYPE=sim
export CHECKPOINT=

python $DIR/Code/train_baseline.py \
	--model_name $DIR/Transformers/${MODEL_NAME} \
	--train_files $DIR/Data/traintest/${FONT_TYPE}/${MODEL_NAME}/SIGHAN/train.txt \
	--val_files $DIR/Data/traintest/${FONT_TYPE}/${MODEL_NAME}/SIGHAN/val.txt \
	--test_files $DIR/csc_evaluation/data/basedata/simplified/test2015.txt \
	--cached_dir $DIR/Cache \
	--result_dir $DIR/Results \
	--glyce_config_path $DIR/Transformers/glyce_bert_both_font.json \
	--vocab_file $DIR/Data/vocab/allNoun.txt \
	--load_pretrain_checkpoint ${CHECKPOINT} \
	--checkpoint_index 19500 \
	--font_type ${FONT_TYPE} \
	--overwrite_cached True \
	--num_train_epochs 5 \
	--gradient_accumulation_steps 2 \
	--use_pinyin True \
	--use_word_feature False \
	--use_copy_label False \
	--compute_metrics True \
	--per_device_train_batch_size 64 \
	--per_device_eval_batch_size 64 \
	--save_steps 500 \
	--logging_steps 500 \
	--fp16 True \
	--do_test False \
