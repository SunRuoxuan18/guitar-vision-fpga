set here [file dirname [file normalize [info script]]]
cd $here
set build "$here/build"
if {[info exists ::env(HAND_CONV_BUILD)]} { set build $::env(HAND_CONV_BUILD) }
open_project -reset $build
set_top hand_conv
add_files "$here/hand_conv.cpp" -cflags "-std=c++14"
add_files -tb "$here/test_hand_conv.cpp" -cflags "-std=c++14"
open_solution -reset solution1 -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
export_design -format ip_catalog -rtl verilog
exit
