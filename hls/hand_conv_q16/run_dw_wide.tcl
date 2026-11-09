set here [file dirname [file normalize [info script]]]
cd $here
open_project -reset "$here/build_dw_wide"
set_top hand_conv_q16
add_files "$here/hand_conv_q16.cpp" -cflags "-std=c++14 -DHAND_POINTWISE8 -DHAND_BURST -DHAND_COMPACT -DHAND_BANKED -DHAND_DW_CACHE -DHAND_PW_FUSED -DHAND_DW_WIDE"
add_files -tb "$here/test_hand_dw_wide.cpp" -cflags "-std=c++14 -DHAND_BURST"
open_solution -reset solution1 -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
remove_files "$here/test_hand_dw_wide.cpp"
add_files -tb "$here/test_hand_dw_interface.cpp" -cflags "-std=c++14 -DHAND_BURST"
csynth_design
cosim_design -rtl verilog -tool xsim
export_design -format ip_catalog -rtl verilog
exit
