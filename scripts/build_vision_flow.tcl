# Combined 640x480 grayscale/Sobel DMA and HLS vision overlay.
#
# Usage:
#   vivado -mode batch -source D:/FPGA/ZynqAudioLab/scripts/build_vision_flow.tcl
#   vivado -mode batch -source D:/FPGA/ZynqAudioLab/scripts/build_vision_flow.tcl -tclargs --prepare-only
#
# Both HLS cores are required by default.  Use --without-lk-pyramid and/or
# --without-roi-projection only for an intentional reduced overlay.  This
# overlay consumes DDR already initialized by Linux and does not initialize PS or DDR.
set root [file dirname [file dirname [file normalize [info script]]]]
set build_dir "$root/diagnostics/vision_flow_build"
set export_dir "$root/output/vision_flow"
if {[info exists ::env(VISION_FLOW_BUILD_DIR)]} { set build_dir [file normalize $::env(VISION_FLOW_BUILD_DIR)] }
if {[info exists ::env(VISION_FLOW_EXPORT_DIR)]} { set export_dir [file normalize $::env(VISION_FLOW_EXPORT_DIR)] }
set prepare_only 0
set reuse 0
set export_only 0
set resume_prepared 0
set include_lk_pyramid 1
set include_roi_projection 1
set lk_ip_dir "$root/hls/lk_pyramid/build/solution1/impl/ip"
set optimized_lk_ip_dir "$root/hls/lk_pyramid/build_optimized/solution1/impl/ip"
if {[file isfile "$optimized_lk_ip_dir/component.xml"]} {
    set lk_ip_dir $optimized_lk_ip_dir
}
if {[info exists ::env(VISION_FLOW_LK_IP_DIR)]} {
    set lk_ip_dir [file normalize $::env(VISION_FLOW_LK_IP_DIR)]
}

foreach argument $argv {
    switch -exact -- $argument {
        --prepare-only { set prepare_only 1 }
        --reuse { set reuse 1 }
        --export-only { set reuse 1; set export_only 1 }
        --resume-prepared { set resume_prepared 1 }
        --without-lk-pyramid { set include_lk_pyramid 0 }
        --without-roi-projection { set include_roi_projection 0 }
        default {
            error "Unknown argument: $argument (use --prepare-only, --reuse, --export-only, --resume-prepared, --without-lk-pyramid, or --without-roi-projection)"
        }
    }
}
if {!$include_lk_pyramid && !$include_roi_projection} {
    error "At least one HLS core must be selected"
}

proc build_prepared_vision_flow {export_dir build_dir} {
    launch_runs synth_1 -jobs 4
    wait_on_run synth_1
    if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
        error "Vision-flow synthesis failed: [get_property STATUS [get_runs synth_1]]"
    }
    launch_runs impl_1 -to_step route_design -jobs 4
    wait_on_run impl_1
    if {![string match "route_design Complete*" [get_property STATUS [get_runs impl_1]]]} {
        error "Vision-flow implementation failed: [get_property STATUS [get_runs impl_1]]"
    }
    open_run impl_1
    set_property BITSTREAM.CONFIG.USERID 32'h56464C31 [current_design]
    report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -file "$export_dir/vision_pl_timing.rpt"
    report_utilization -hierarchical -hierarchical_depth 2 -file "$export_dir/vision_pl_utilization.rpt"
    report_drc -file "$export_dir/vision_pl_drc.rpt"
    report_route_status -file "$export_dir/vision_pl_route.rpt"
    report_methodology -file "$export_dir/vision_pl_methodology.rpt"
    set setup_paths [get_timing_paths -delay_type max -max_paths 1]
    set hold_paths [get_timing_paths -delay_type min -max_paths 1]
    if {[llength $setup_paths] == 0 || [llength $hold_paths] == 0} { error "Vision-flow timing is missing constrained setup/hold paths" }
    set setup_slack [get_property SLACK [lindex $setup_paths 0]]
    set hold_slack [get_property SLACK [lindex $hold_paths 0]]
    if {$setup_slack < 0 || $hold_slack < 0} { error "Vision-flow timing failed: setup=$setup_slack ns hold=$hold_slack ns" }
    set drc_failures [get_drc_violations -quiet -filter {SEVERITY == Error || SEVERITY == {Critical Warning}}]
    if {[llength $drc_failures] != 0} { error "Vision-flow DRC failed: $drc_failures" }
    write_checkpoint -force "$export_dir/vision_pl_routed.dcp"
    write_bitstream -force "$export_dir/vision_pl.bit"
    set hwh "$build_dir/vision_pl.gen/sources_1/bd/vision_pl/hw_handoff/vision_pl.hwh"
    if {![file isfile $hwh]} { error "Missing generated vision hardware handoff: $hwh" }
    file copy -force $hwh "$export_dir/vision_pl.hwh"
    puts "VISION_FLOW_BUILD_PASS: setup=$setup_slack ns hold=$hold_slack ns USERID=0x56464C31"
    puts "VISION_FLOW_OUTPUT: $export_dir/vision_pl.bit $export_dir/vision_pl.hwh"
}

if {$resume_prepared} {
    if {$prepare_only || $reuse || $export_only} { error "--resume-prepared cannot be combined with --prepare-only, --reuse, or --export-only" }
    file mkdir $export_dir
    set project "$build_dir/vision_pl.xpr"
    if {![file isfile $project]} { error "No prepared vision-flow project: $project" }
    cd $build_dir
    set_param general.maxThreads 4
    open_project $project
    puts "VISION_FLOW_RESUMED: $project"
    build_prepared_vision_flow $export_dir $build_dir
    close_project
    exit 0
}

# Preflight before starting the child Vivado process that creates the base
# vision project.  A missing HLS export must never trigger a partial build.
set core_specs {}
if {$include_lk_pyramid} {
    lappend core_specs [list lk_pyramid 0x43c00000 $lk_ip_dir]
}
if {$include_roi_projection} {
    lappend core_specs [list roi_projection 0x43c10000 "$root/hls/roi_projection/build/solution1/impl/ip"]
}
foreach spec $core_specs {
    lassign $spec top base ip_dir
    if {![file isdirectory $ip_dir] || [llength [glob -nocomplain -directory $ip_dir component.xml]] == 0} {
        error "Build/export $top HLS IP before running this flow: $ip_dir/component.xml"
    }
}

file mkdir $build_dir
file mkdir $export_dir

set project "$build_dir/vision_pl.xpr"
# Build the known-good DMA/stream base once in this flow's isolated directory.
# On --reuse the project already exists, so reopening it directly avoids
# rerunning the base flow's module-reference refresh.
if {!$reuse} {
    set ::env(VISION_PL_BUILD_DIR) $build_dir
    set ::env(VISION_PL_EXPORT_DIR) $export_dir
    if {[catch {
        exec [info nameofexecutable] -mode batch -source "$root/scripts/build_vision_pl.tcl" -tclargs --prepare-only
    } base_output]} {
        error "Base vision prepare failed:\n$base_output"
    }
    puts $base_output
}
if {![file isfile $project]} { error "Base vision project was not created: $project" }
cd $build_dir
set_param general.maxThreads 4
open_project $project
if {[get_property PART [current_project]] ne "xc7z020clg400-1"} {
    error "Unexpected part in base vision project"
}

set ip_repos {}
foreach spec $core_specs { lappend ip_repos [lindex $spec 2] }
set_property ip_repo_paths $ip_repos [current_project]
update_ip_catalog
open_bd_design [get_files vision_pl.bd]

set core_cells {}
set all_masters {}
foreach spec $core_specs {
    lassign $spec top base ip_dir
    set defs [get_ipdefs -all -filter "NAME == $top"]
    if {[llength $defs] != 1} {
        error "Expected exactly one IP definition named $top in $ip_dir; found: $defs"
    }
    set core [create_bd_cell -type ip -vlnv [lindex $defs 0] $top]
    # HLS interface VLNVs vary by tool release.  Enumerate the instantiated
    # core's interface pins and classify their explicit MODE property instead.
    set controls {}
    set masters {}
    foreach intf [get_bd_intf_pins -quiet -of_objects $core] {
        switch -- [get_property MODE $intf] {
            Slave { lappend controls $intf }
            Master { lappend masters $intf }
        }
    }
    if {[llength $controls] != 1 || [llength $masters] < 1} {
        error "Unexpected $top AXI interface topology: controls=$controls masters=$masters"
    }
    puts "VISION_FLOW_HLS_INTERFACES: $top control=$controls masters=$masters"
    lappend core_cells [list $core $base $controls $masters]
    foreach master $masters { lappend all_masters [list $core $master] }
}

set control_axi [get_bd_cells control_axi]
set memory_axi [get_bd_cells memory_axi]
set control_mi_start [get_property CONFIG.NUM_MI $control_axi]
set memory_si_start [get_property CONFIG.NUM_SI $memory_axi]
set_property CONFIG.NUM_MI [expr {$control_mi_start + [llength $core_cells]}] $control_axi
set_property CONFIG.NUM_SI [expr {$memory_si_start + [llength $all_masters]}] $memory_axi

set control_index $control_mi_start
foreach core_info $core_cells {
    lassign $core_info core base controls masters
    connect_bd_intf_net [get_bd_intf_pins [format "control_axi/M%02d_AXI" $control_index]] [lindex $controls 0]
    incr control_index
}
set memory_index $memory_si_start
foreach master_info $all_masters {
    lassign $master_info core master
    connect_bd_intf_net $master [get_bd_intf_pins [format "memory_axi/S%02d_AXI" $memory_index]]
    incr memory_index
}

# Clock and reset only the ports added above; the base script already owns the
# DMA and original interconnect ports.  HLS reset ports must be active-low.
set fclk [get_bd_pins ps7/FCLK_CLK0]
set aresetn [get_bd_pins reset_control/peripheral_aresetn]
foreach core_info $core_cells {
    lassign $core_info core base controls masters
    set core_clocks [get_bd_pins "$core/*" -filter {TYPE == clk}]
    set core_resets [get_bd_pins "$core/*" -filter {TYPE == rst}]
    if {[llength $core_clocks] == 0 || [llength $core_resets] == 0} {
        error "HLS core $core must expose clock and reset pins"
    }
    foreach reset $core_resets {
        set polarity [get_property CONFIG.POLARITY $reset]
        if {$polarity ne "" && $polarity ne "ACTIVE_LOW"} {
            error "HLS reset $reset is not active-low: $polarity"
        }
    }
    foreach clock $core_clocks { connect_bd_net $fclk $clock }
    foreach reset $core_resets { connect_bd_net $aresetn $reset }
}
for {set index $control_mi_start} {$index < $control_index} {incr index} {
    connect_bd_net $fclk [get_bd_pins [format "control_axi/M%02d_ACLK" $index]]
    connect_bd_net $aresetn [get_bd_pins [format "control_axi/M%02d_ARESETN" $index]]
}
for {set index $memory_si_start} {$index < $memory_index} {incr index} {
    connect_bd_net $fclk [get_bd_pins [format "memory_axi/S%02d_ACLK" $index]]
    connect_bd_net $aresetn [get_bd_pins [format "memory_axi/S%02d_ARESETN" $index]]
}

set hp0_segment [get_bd_addr_segs ps7/S_AXI_HP0/HP0_DDR_LOWOCM]
foreach core_info $core_cells {
    lassign $core_info core base controls masters
    set control_segments [get_bd_addr_segs -quiet -of_objects [lindex $controls 0]]
    if {[llength $control_segments] != 1} {
        error "Cannot resolve control address segment for [lindex $controls 0]: $control_segments"
    }
    assign_bd_address -offset $base -range 0x00010000 \
        -target_address_space [get_bd_addr_spaces ps7/Data] \
        [lindex $control_segments 0] -force
    foreach master $masters {
        set space [get_bd_addr_spaces -quiet [format "%s/Data_%s" $core [get_property NAME $master]]]
        if {[llength $space] != 1} {
            error "Cannot resolve HLS address space for $master: $space"
        }
        assign_bd_address -offset 0x00000000 -range 0x20000000 \
            -target_address_space $space $hp0_segment -force
    }
}

validate_bd_design
save_bd_design
set ps [get_bd_cells ps7]
if {[get_property CONFIG.PCW_EN_DDR $ps] != 1 ||
    [get_property CONFIG.PCW_S_AXI_HP0_DATA_WIDTH $ps] != 64 ||
    abs([get_property CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ $ps] - 100.0) > 0.01} {
    error "Vision flow must retain DDR decoding, 64-bit HP0, and FCLK0 at 100 MHz"
}
if {[llength [get_bd_intf_ports -quiet -filter {VLNV =~ *:ddrx:*}]] != 0 ||
    [llength [get_bd_intf_ports -quiet]] != 1} {
    error "Vision flow must expose only FIXED_IO, with no PL or DDR pins"
}

set bd_file [get_files vision_pl.bd]
generate_target all $bd_file
make_wrapper -files $bd_file -top
set_property top vision_pl_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1
write_bd_tcl -force "$build_dir/vision_flow_bd.tcl"
puts "VISION_FLOW_CONFIG: FCLK0=100MHz HP0=64 DMA=0x40400000 LK=0x43c00000 ROI=0x43c10000 LK_IP=$lk_ip_dir"
puts "VISION_FLOW_LINUX_DDR: existing PS DDR initialization preserved; no PS or DDR initialization is run"
if {$prepare_only} {
    puts "VISION_FLOW_PREPARED: $project"
    close_project
    exit 0
}

if {$reuse && !$export_only} {
    reset_run impl_1
    reset_run synth_1
}
if {!$export_only} {
    launch_runs synth_1 -jobs 4
    wait_on_run synth_1
}
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
    error "Vision-flow synthesis failed: [get_property STATUS [get_runs synth_1]]"
}
if {!$export_only} {
    launch_runs impl_1 -to_step route_design -jobs 4
    wait_on_run impl_1
}
if {![string match "route_design Complete*" [get_property STATUS [get_runs impl_1]]]} {
    error "Vision-flow implementation failed: [get_property STATUS [get_runs impl_1]]"
}
open_run impl_1
set_property BITSTREAM.CONFIG.USERID 32'h56464C31 [current_design]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
    -file "$export_dir/vision_pl_timing.rpt"
report_utilization -hierarchical -hierarchical_depth 2 -file "$export_dir/vision_pl_utilization.rpt"
report_drc -file "$export_dir/vision_pl_drc.rpt"
report_route_status -file "$export_dir/vision_pl_route.rpt"
report_methodology -file "$export_dir/vision_pl_methodology.rpt"
set setup_paths [get_timing_paths -delay_type max -max_paths 1]
set hold_paths [get_timing_paths -delay_type min -max_paths 1]
if {[llength $setup_paths] == 0 || [llength $hold_paths] == 0} {
    error "Vision-flow timing is missing constrained setup/hold paths"
}
set setup_slack [get_property SLACK [lindex $setup_paths 0]]
set hold_slack [get_property SLACK [lindex $hold_paths 0]]
if {$setup_slack < 0 || $hold_slack < 0} {
    error "Vision-flow timing failed: setup=$setup_slack ns hold=$hold_slack ns"
}
set drc_failures [get_drc_violations -quiet -filter {SEVERITY == Error || SEVERITY == {Critical Warning}}]
if {[llength $drc_failures] != 0} { error "Vision-flow DRC failed: $drc_failures" }
write_checkpoint -force "$export_dir/vision_pl_routed.dcp"
write_bitstream -force "$export_dir/vision_pl.bit"
set hwh "$build_dir/vision_pl.gen/sources_1/bd/vision_pl/hw_handoff/vision_pl.hwh"
if {![file isfile $hwh]} { error "Missing generated vision hardware handoff: $hwh" }
file copy -force $hwh "$export_dir/vision_pl.hwh"
puts "VISION_FLOW_BUILD_PASS: setup=$setup_slack ns hold=$hold_slack ns USERID=0x56464C31"
puts "VISION_FLOW_RESOURCE_NOTE: inspect $export_dir/vision_pl_utilization.rpt; xc7z020 has 140 BRAM36 tiles"
puts "VISION_FLOW_OUTPUT: $export_dir/vision_pl.bit $export_dir/vision_pl.hwh"
close_project
