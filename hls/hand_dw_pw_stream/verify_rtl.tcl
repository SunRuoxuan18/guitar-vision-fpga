set here [file dirname [file normalize [info script]]]
cd $here
open_project "$here/build"
open_solution solution1
cosim_design -rtl verilog -tool xsim
export_design -format ip_catalog -rtl verilog
exit
