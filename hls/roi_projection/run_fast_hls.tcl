set here [file dirname [file normalize [info script]]]
open_project -reset "$here/build_fast"
set_top roi_projection
add_files "$here/roi_projection_fast.cpp"
add_files -tb "$here/test_roi_projection.cpp" -cflags "-std=c++14"
open_solution -reset solution1 -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
export_design -format ip_catalog -rtl verilog
exit
