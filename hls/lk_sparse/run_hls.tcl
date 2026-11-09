set here [file dirname [file normalize [info script]]]
open_project -reset "$here/build"
set_top lk_sparse
add_files "$here/lk_sparse.cpp"
add_files -tb "$here/test_lk_sparse.cpp" -cflags "-std=c++14"
open_solution -reset solution1 -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
if {[info exists ::env(LK_HLS_COSIM)] && $::env(LK_HLS_COSIM) eq "1"} {
    cosim_design -rtl verilog
}
export_design -format ip_catalog -rtl verilog
exit
