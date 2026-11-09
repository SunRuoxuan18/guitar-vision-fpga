# Prepare an isolated hand-neural clone of the routed AVS1 block design.
# This script does not synthesize or implement.  It requires the exported HLS
# IP at hls/hand_conv/build/solution1/impl/ip/component.xml.
#
# Usage:
#   vivado -mode batch -source D:/FPGA/ZynqAudioLab/scripts/build_hand_neural.tcl -tclargs --prepare-only

set root [file dirname [file dirname [file normalize [info script]]]]
set source_build "$root/diagnostics/av_system_build"
set clone_build "$root/diagnostics/hand_neural_build"
set out "$root/output/hand_neural"
set source_project "$source_build/vision_pl.xpr"
set hls_ip "$root/hls/hand_conv/build/solution1/impl/ip"
set hand_name hand_conv
if {[info exists ::env(HAND_NEURAL_FAST)] && $::env(HAND_NEURAL_FAST) eq "1"} {
    set clone_build "$root/diagnostics/hand_neural_fast_build"
    set out "$root/output/hand_neural_fast"
    set hls_ip "$root/hls/hand_conv/build_fast/solution1/impl/ip"
}
if {[info exists ::env(HAND_NEURAL_BUFFERED)] && $::env(HAND_NEURAL_BUFFERED) eq "1"} {
    set clone_build "$root/diagnostics/hand_neural_buffered_build"
    set out "$root/output/hand_neural_buffered"
    set hls_ip "$root/hls/hand_conv/build_buffered/solution1/impl/ip"
}
if {[info exists ::env(HAND_NEURAL_Q16)] && $::env(HAND_NEURAL_Q16) eq "1"} {
    set clone_build "$root/diagnostics/hand_neural_q16_build"
    set out "$root/output/hand_neural_q16"
    set hls_ip "$root/hls/hand_conv_q16/build/solution1/impl/ip"
    set hand_name hand_conv_q16
}
set hand_base 0x43C30000
if {[info exists ::env(HAND_NEURAL_REALTIME)] && $::env(HAND_NEURAL_REALTIME) eq "1"} {
    set clone_build "$root/diagnostics/hand_neural_realtime_v3_build"
    set out "$root/output/hand_neural_realtime"
    set hls_ip "$root/hls/hand_conv_q16/build_realtime/solution1/impl/ip"
    set hand_name hand_conv_q16
}

set prepare_only 0
if {[info exists ::env(HAND_NEURAL_BURST)] && $::env(HAND_NEURAL_BURST) eq "1"} {
    set clone_build "$root/diagnostics/hand_neural_burst_build"
    set out "$root/output/hand_neural_burst"
    set hls_ip "$root/hls/hand_conv_q16/build_burst/solution1/impl/ip"
    set hand_name hand_conv_q16
}
if {[info exists ::env(HAND_NEURAL_COMPACT)] && $::env(HAND_NEURAL_COMPACT) eq "1"} {
    set clone_build "$root/diagnostics/hand_neural_compact_build"
    set out "$root/output/hand_neural_compact"
    set hls_ip "$root/hls/hand_conv_q16/build_compact/solution1/impl/ip"
    set hand_name hand_conv_q16
}
if {[info exists ::env(HAND_NEURAL_BANKED)] && $::env(HAND_NEURAL_BANKED) eq "1"} {
    set clone_build "$root/diagnostics/hand_neural_banked_build"
    set out "$root/output/hand_neural_banked"
    set hls_ip "$root/hls/hand_conv_q16/build_banked/solution1/impl/ip"
    set hand_name hand_conv_q16
}
if {[info exists ::env(HAND_NEURAL_LEAN)] && $::env(HAND_NEURAL_LEAN) eq "1"} {
    set clone_build "$root/diagnostics/hand_neural_lean_build"
    set out "$root/output/hand_neural_lean"
    set hls_ip "$root/hls/hand_conv_q16/build_lean/solution1/impl/ip"
    set hand_name hand_conv_q16
}
if {[info exists ::env(HAND_NEURAL_LEAN_IO)] && $::env(HAND_NEURAL_LEAN_IO) eq "1"} {
    set clone_build "$root/diagnostics/hand_neural_lean_io_build"
    set out "$root/output/hand_neural_lean_io"
    set hls_ip "$root/hls/hand_conv_q16/build_lean_io/solution1/impl/ip"
    set hand_name hand_conv_q16
}
if {[info exists ::env(HAND_NEURAL_DW_BOUNDED)] && $::env(HAND_NEURAL_DW_BOUNDED) eq "1"} {
    set clone_build "$root/diagnostics/hand_neural_dw_bounded_build"
    set out "$root/output/hand_neural_dw_bounded"
    set hls_ip "$root/hls/hand_conv_q16/build_depthwise_bounded/solution1/impl/ip"
    set hand_name hand_conv_q16
}
if {[info exists ::env(HAND_NEURAL_PW_CACHED)] && $::env(HAND_NEURAL_PW_CACHED) eq "1"} {
    set clone_build "$root/diagnostics/hand_neural_pw_cached_build"
    set out "$root/output/hand_neural_pw_cached"
    set hls_ip "$root/hls/hand_conv_q16/build_pointwise_cached/solution1/impl/ip"
    set hand_name hand_conv_q16
}
if {[info exists ::env(HAND_NEURAL_FUSED)] && $::env(HAND_NEURAL_FUSED) eq "1"} {
    set clone_build "$root/diagnostics/hand_neural_fused_build"
    set out "$root/output/hand_neural_fused"
    set hls_ip "$root/hls/hand_conv_q16/build_fused/solution1/impl/ip"
    set hand_name hand_conv_q16
}
if {[info exists ::env(HAND_NEURAL_DW_WIDE)] && $::env(HAND_NEURAL_DW_WIDE) eq "1"} {
    set clone_build "$root/diagnostics/hand_neural_dw_wide_build"
    set out "$root/output/hand_neural_dw_wide"
    set hls_ip "$root/hls/hand_conv_q16/build_dw_wide/solution1/impl/ip"
    set hand_name hand_conv_q16
}
foreach argument $argv {
    switch -exact -- $argument {
        --prepare-only { set prepare_only 1 }
        default { error "Unknown argument: $argument (use --prepare-only)" }
    }
}

if {![file isfile $source_project]} {
    error "Known AVS1 source project is missing: $source_project"
}
if {![file isfile "$hls_ip/component.xml"]} {
    error "hand_conv HLS export is not ready: $hls_ip/component.xml"
}
if {[file exists $clone_build]} {
    error "Refusing to overwrite existing isolated build: $clone_build"
}
file mkdir $out

open_project $source_project
save_project_as -include_local_ip_cache vision_pl $clone_build
close_project
open_project "$clone_build/vision_pl.xpr"
cd $clone_build
set_param general.maxThreads 1

set ip_repos [get_property ip_repo_paths [current_project]]
if {[lsearch -exact $ip_repos $hls_ip] < 0} { lappend ip_repos $hls_ip }
set_property ip_repo_paths $ip_repos [current_project]
update_ip_catalog
set bd_path [file normalize [get_files vision_pl.bd]]
if {![string equal -nocase [string range $bd_path 0 [expr {[string length $clone_build]-1}]] $clone_build]} {
    error "Cloned BD is not isolated: $bd_path"
}
open_bd_design [get_files vision_pl.bd]

set defs [get_ipdefs -all -filter "NAME == $hand_name"]
if {[llength $defs] != 1} {
    error "Expected exactly one hand_conv IP definition in $hls_ip; found: $defs"
}
set hand [create_bd_cell -type ip -vlnv [lindex $defs 0] $hand_name]

set controls {}
set masters {}
foreach intf [get_bd_intf_pins -quiet -of_objects $hand] {
    switch -- [get_property MODE $intf] {
        Slave { lappend controls $intf }
        Master { lappend masters $intf }
    }
}
if {[llength $controls] != 1 || [llength $masters] ni {1 2}} {
    error "Unexpected hand_conv AXI topology: controls=$controls masters=$masters"
}
foreach master $masters {
    if {[get_property NAME $master] ni {m_axi_gmem m_axi_results}} {
        error "Unexpected hand_conv master interface: $master"
    }
}
if {[get_property NAME [lindex $controls 0]] ne "s_axi_control"} {
    error "Expected hand_conv control interface s_axi_control: $controls"
}

set control_axi [get_bd_cells control_axi]
set memory_axi [get_bd_cells memory_axi]
set control_index [get_property CONFIG.NUM_MI $control_axi]
set memory_index [get_property CONFIG.NUM_SI $memory_axi]
set_property CONFIG.NUM_MI [expr {$control_index + 1}] $control_axi
set_property CONFIG.NUM_SI [expr {$memory_index + [llength $masters]}] $memory_axi

connect_bd_intf_net [get_bd_intf_pins [format "control_axi/M%02d_AXI" $control_index]] [lindex $controls 0]
set master_index $memory_index
foreach master $masters {
    connect_bd_intf_net $master [get_bd_intf_pins [format "memory_axi/S%02d_AXI" $master_index]]
    incr master_index
}

set fclk [get_bd_pins ps7/FCLK_CLK0]
set aresetn [get_bd_pins reset_control/peripheral_aresetn]
set clocks [get_bd_pins "$hand/*" -filter {TYPE == clk}]
set resets [get_bd_pins "$hand/*" -filter {TYPE == rst}]
if {[llength $clocks] == 0 || [llength $resets] == 0} {
    error "hand_conv does not expose clock/reset pins: clocks=$clocks resets=$resets"
}
foreach reset $resets {
    set polarity [get_property CONFIG.POLARITY $reset]
    if {$polarity ne "" && $polarity ne "ACTIVE_LOW"} {
        error "hand_conv reset is not active-low: $reset polarity=$polarity"
    }
}
foreach clock $clocks { connect_bd_net $fclk $clock }
foreach reset $resets { connect_bd_net $aresetn $reset }
connect_bd_net $fclk [get_bd_pins [format "control_axi/M%02d_ACLK" $control_index]]
connect_bd_net $aresetn [get_bd_pins [format "control_axi/M%02d_ARESETN" $control_index]]
for {set index $memory_index} {$index < $master_index} {incr index} {
    connect_bd_net $fclk [get_bd_pins [format "memory_axi/S%02d_ACLK" $index]]
    connect_bd_net $aresetn [get_bd_pins [format "memory_axi/S%02d_ARESETN" $index]]
}

set hp0_segment [get_bd_addr_segs ps7/S_AXI_HP0/HP0_DDR_LOWOCM]
set control_segments [get_bd_addr_segs -quiet -of_objects [lindex $controls 0]]
if {[llength $control_segments] != 1} {
    error "Could not resolve hand_conv control segment: $control_segments"
}
assign_bd_address -offset $hand_base -range 0x00010000 \
    -target_address_space [get_bd_addr_spaces ps7/Data] [lindex $control_segments 0] -force
foreach master $masters {
    set memory_space [get_bd_addr_spaces -quiet [format "%s/Data_%s" $hand_name [get_property NAME $master]]]
    if {[llength $memory_space] != 1} { error "Could not resolve hand_conv memory space: $master" }
    assign_bd_address -offset 0x00000000 -range 0x20000000 \
        -target_address_space $memory_space $hp0_segment -force
}

validate_bd_design
generate_target all [get_files vision_pl.bd]
save_bd_design
write_bd_tcl -force "$out/hand_neural_bd.tcl"

set audio_clk [get_bd_pins ps7/FCLK_CLK1]
if {[llength $audio_clk] != 1 || [get_property CONFIG.FREQ_HZ $audio_clk] != 125000000} {
    error "Audio clock was not preserved at 125 MHz: $audio_clk"
}
puts "HAND_NEURAL_PREPARED: project=$clone_build/vision_pl.xpr control=0x43C30000 memory=ps7/S_AXI_HP0 hand_clk=100MHz audio_clk=125MHz"
close_project
