set here [file dirname [file normalize [info script]]]
if {[llength $argv] != 1} { error "Expected a completed HLS probe name" }
set variant [lindex $argv 0]
if {![regexp {^[a-zA-Z0-9_]+$} $variant]} { error "Invalid probe name" }
set build [file normalize "$here/../../diagnostics/hand_conv_preopt_20261028/$variant"]
set rtl "$build/solution1/syn/verilog"
set out "$build/ooc"
if {![file isfile "$rtl/hand_conv_preopt.v"]} { error "HLS synthesis is missing" }
if {[file exists $out]} { error "Refusing to overwrite OOC result $out" }
file mkdir $out
cd $out
set_param general.maxThreads 2
read_verilog [glob "$rtl/*.v"]
synth_design -mode out_of_context -top hand_conv_preopt -part xc7z020clg400-1 -flatten_hierarchy rebuilt
create_clock -name ap_clk -period 10 [get_ports ap_clk]
report_utilization -file "$out/utilization.rpt"
report_utilization -hierarchical -file "$out/utilization_hierarchical.rpt"
report_timing_summary -file "$out/timing_synth.rpt"
write_checkpoint "$out/synth.dcp"
puts "PREOPT_OOC_SYNTH_PASS: $variant"
