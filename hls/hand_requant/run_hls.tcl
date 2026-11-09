set here [file dirname [file normalize [info script]]]
cd $here
open_project -reset "$here/build"
set_top hand_requant
add_files "$here/hand_requant.cpp" -cflags "-std=c++14"
add_files -tb "$here/test_hand_requant.cpp" -cflags "-std=c++14"
open_solution -reset solution1 -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
cosim_design -rtl verilog -tool xsim
export_design -format ip_catalog -rtl verilog
exit
