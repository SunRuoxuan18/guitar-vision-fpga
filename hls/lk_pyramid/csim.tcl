set here [file dirname [file normalize [info script]]]
open_project -reset "$here/csim_build"
set_top lk_pyramid
add_files "$here/lk_pyramid.cpp"
add_files -tb "$here/test_lk_pyramid.cpp" -cflags "-std=c++14"
add_files -tb "$here/cpu_lk_pyramid.cpp" -cflags "-std=c++14"
open_solution -reset solution1 -flow_target vivado
set_part {xc7z020clg400-1}
csim_design
exit
