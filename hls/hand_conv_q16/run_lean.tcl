set here [file dirname [file normalize [info script]]]
cd $here
open_project -reset "$here/build_lean"
set_top hand_conv_q16
add_files "$here/hand_conv_q16.cpp" -cflags "-std=c++14 -DHAND_POINTWISE8 -DHAND_BURST -DHAND_COMPACT -DHAND_BANKED"
add_files -tb "$here/test_hand_conv_q16.cpp" -cflags "-std=c++14 -DHAND_BURST"
open_solution -reset solution1 -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
export_design -format ip_catalog -rtl verilog
exit
