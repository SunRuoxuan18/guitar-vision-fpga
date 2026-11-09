set here [file dirname [file normalize [info script]]]
cd $here
open_project -reset "$here/build"
set_top hand_dw_pw_stream
add_files "$here/hand_dw_pw_stream.cpp" -cflags "-std=c++14 -ffp-contract=off"
add_files -tb "$here/test_hand_dw_pw_stream.cpp" -cflags "-std=c++14 -ffp-contract=off"
open_solution -reset solution1 -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
if {[info exists ::env(HAND_STREAM_COSIM)] && $::env(HAND_STREAM_COSIM) eq "1"} {
    cosim_design -rtl verilog -tool xsim
    export_design -format ip_catalog -rtl verilog
}
exit
