set here [file dirname [file normalize [info script]]]
cd $here
open_project "$here/build_burst"
open_solution solution1
csim_design
exit
