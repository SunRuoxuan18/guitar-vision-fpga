set here [file dirname [file normalize [info script]]]
open_project "$here/build"
open_solution solution1
csim_design
cosim_design -rtl verilog
export_design -format ip_catalog -rtl verilog
exit
