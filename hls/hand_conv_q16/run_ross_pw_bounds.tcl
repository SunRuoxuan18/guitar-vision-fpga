set here [file dirname [file normalize [info script]]]
cd $here
set variant $::env(ROSS_PW_VARIANT)
if {$variant ni {baseline candidate first_fold}} { error "Expected baseline, candidate or first_fold" }
set flags "-std=c++14 -DHAND_POINTWISE8 -DHAND_BURST -DHAND_COMPACT -DHAND_BANKED -DHAND_DW_CACHE -DHAND_PW_FUSED -DHAND_DW_WIDE"
if {$variant eq "candidate"} { append flags " -DHAND_PW_BOUNDED" }
if {$variant eq "first_fold"} { append flags " -DHAND_PW_FIRST_FOLD" }
if {[info exists ::env(ROSS_PW_VERIFY_ONLY)] && $::env(ROSS_PW_VERIFY_ONLY) eq "1"} {
    open_project "$here/build_ross_pw_$variant"
    open_solution solution1
    csim_design
    cosim_design -rtl verilog -tool xsim -argv "bounded-smoke"
    exit
}
open_project -reset "$here/build_ross_pw_$variant"
set_top hand_conv_q16
add_files "$here/hand_conv_q16.cpp" -cflags $flags
add_files -tb "$here/test_hand_pw_bounds.cpp" -cflags "-std=c++14 -DHAND_BURST"
open_solution -reset solution1 -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
cosim_design -rtl verilog -tool xsim -argv "bounded-smoke"
export_design -format ip_catalog -rtl verilog
exit
