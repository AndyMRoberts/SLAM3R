#!/bin/bash


TEST_DATASET="/home/campus.ncl.ac.uk/c4071391/datasets/agricultural/LettuceForSLAM/IEEE Lettuce Farm SLAM Dataset/Sequence_9/_2023-05-03-10-27-15/D435i/RGB/"


######################################################################################
# set the parameters for whole scene reconstruction below
# for defination of these parameters, please refer to the recon.py
######################################################################################
TEST_NAME="Replica_demo_LFSD"
KEYFRAME_STRIDE=20
UPDATE_BUFFER_INTV=3
MAX_NUM_REGISTER=10
WIN_R=5
NUM_SCENE_FRAME=10
INITIAL_WINSIZE=5 
CONF_THRES_L2W=10
CONF_THRES_I2P=1.5
NUM_POINTS_SAVE=1000000

GPU_ID=-1


python recon.py \
--test_name $TEST_NAME \
--dataset "${TEST_DATASET}" \
--gpu_id $GPU_ID \
--keyframe_stride $KEYFRAME_STRIDE \
--win_r $WIN_R \
--num_scene_frame $NUM_SCENE_FRAME \
--initial_winsize $INITIAL_WINSIZE \
--conf_thres_l2w $CONF_THRES_L2W \
--conf_thres_i2p $CONF_THRES_I2P \
--num_points_save $NUM_POINTS_SAVE \
--update_buffer_intv $UPDATE_BUFFER_INTV \
--max_num_register $MAX_NUM_REGISTER \
# --online \
--retrieve_freq 10 \
