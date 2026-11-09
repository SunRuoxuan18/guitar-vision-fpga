set here [file dirname [file normalize [info script]]]
set probe_root [file normalize "$here/../../diagnostics/yolo_q16_shape_probe"]
set project_name [clock format [clock seconds] -format {hls_%Y%m%d_%H%M%S}]
set project_path "$probe_root/$project_name"
file mkdir $probe_root
cd $here
open_project $project_path
set_top hand_conv_q16
set flags "-std=c++14 -DHAND_EXPERIMENTAL_YOLO320 -DHAND_POINTWISE8 -DHAND_BURST -DHAND_COMPACT -DHAND_BANKED -DHAND_DW_CACHE -DHAND_PW_FUSED -DHAND_DW_WIDE -DHAND_PW_FIRST_FOLD"
add_files "$here/hand_conv_q16.cpp" -cflags $flags
add_files -tb "$here/test_hand_conv_q16.cpp" -cflags "-std=c++14 -DHAND_BURST -DHAND_EXPERIMENTAL_YOLO320"
open_solution solution1 -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
exit
