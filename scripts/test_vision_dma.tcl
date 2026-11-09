# Standalone: D:/AMD/2026.1/Vivado/bin/xtclsh.bat <this script> [--synth]
# Runs isolated XSim tests, including unchanged existing vision tests. Optional
# synthesis checks default geometry and Vivado module-reference AXI inference.
# Generated artifacts stay in TEMP; no existing Vivado project is modified.
set dma_root [file dirname [file dirname [file normalize [info script]]]]
set dma_bin "D:/AMD/2026.1/Vivado/bin"
if {[info exists ::env(VIVADO_BIN)]} { set dma_bin $::env(VIVADO_BIN) }
set dma_started [clock milliseconds]
set dma_deadline [expr {$dma_started + 600000}]
set dma_original_dir [pwd]
set dma_temp [file join $::env(TEMP) "vision_dma_[pid]_[clock seconds]"]
file mkdir $dma_temp
puts "Vision DMA logs and generated artifacts: $dma_temp"

proc dma_run_tool {name args} {
    global dma_bin dma_deadline
    if {[clock milliseconds] >= $dma_deadline} { error "Vision DMA verification deadline exceeded" }
    set executable [file join $dma_bin "${name}.bat"]
    if {![file exists $executable]} { error "Missing Vivado tool: $executable" }
    set command [list $::env(COMSPEC) /c $executable {*}$args]
    puts "Running $name $args"
    set channel [open [concat [list |] $command [list 2>@1]] r]
    fconfigure $channel -blocking 0 -buffering none
    set output ""
    while {1} {
        set chunk [read $channel]
        append output $chunk
        puts -nonewline $chunk
        flush stdout
        if {[eof $channel]} { break }
        if {[clock milliseconds] >= $dma_deadline} {
            foreach child [pid $channel] { catch {exec taskkill /PID $child /T /F 2>@1} }
            catch {close $channel}
            error "Vision DMA verification exceeded 600 seconds while running $name"
        }
        after 20
    }
    fconfigure $channel -blocking 1
    if {[catch {close $channel} message]} { error "$name failed: $message\n$output" }
    return $output
}

set dma_failed [catch {
    cd $dma_temp
    dma_run_tool xvlog -sv [file join $dma_root rtl vision_rgb_gray.v] \
        [file join $dma_root rtl vision_sobel3x3.v] \
        [file join $dma_root rtl vision_dma_stream.v] \
        [file join $dma_root sim tb_vision_dma_stream.v] \
        [file join $dma_root sim tb_vision_preprocess.v] -log vision_dma_compile.log
    foreach {top snapshot marker} {
        tb_vision_dma_stream vision_dma_w7_h5 {VISION_DMA_PASS width=7 height=5 }
        tb_vision_dma_stream_min3 vision_dma_w3_h3 {VISION_DMA_PASS width=3 height=3 }
        tb_vision_dma_stream_row1 vision_dma_w5_h1 {VISION_DMA_PASS width=5 height=1 }
        tb_vision_dma_stream_vga vision_dma_w640_h480 {VISION_DMA_PASS width=640 height=480 }
        tb_vision_preprocess vision_preprocess_w5 {VISION_PREPROCESS_PASS width=5 }
        tb_vision_preprocess_min3 vision_preprocess_w3 {VISION_PREPROCESS_PASS width=3 }
    } {
        dma_run_tool xelab $top -s $snapshot -debug typical -log "${snapshot}_elab.log"
        set result [dma_run_tool xsim $snapshot -runall -onerror quit -onfinish quit \
            -log "${snapshot}_sim.log"]
        if {![string match "*$marker*" $result]} { error "Missing successful test result: $marker" }
    }
    puts "VISION_DMA_SIMULATION_PASS: byte-exact 7x5, 3x3, 5x1, 640x480; existing width-5/3 tests pass"

    if {[info exists argv] && [lsearch -exact $argv --synth] >= 0} {
        set synthesis_script [file join $dma_temp synthesis.tcl]
        set channel [open $synthesis_script w]
        puts $channel {create_project -in_memory -part xc7z020clg400-1}
        puts $channel {set_property source_mgmt_mode All [current_project]}
        puts $channel {set_property top vision_dma_stream [get_filesets sources_1]}
        foreach rtl_file {vision_rgb_gray.v vision_sobel3x3.v vision_dma_stream.v} {
            puts $channel [list add_files [file join $dma_root rtl $rtl_file]]
        }
        puts $channel {update_compile_order -fileset sources_1}
        puts $channel {create_bd_design dma_interface_check}
        puts $channel {create_bd_cell -type module -reference vision_dma_stream vision_dma}
        puts $channel {
            foreach {name mode} {S_AXIS Slave M_AXIS Master} {
                set intf [get_bd_intf_pins vision_dma/$name]
                if {[llength $intf] != 1 || [get_property MODE $intf] ne $mode ||
                    [get_property CONFIG.TDATA_NUM_BYTES $intf] != 4 ||
                    [get_property CONFIG.HAS_TKEEP $intf] != 1 ||
                    [get_property CONFIG.HAS_TLAST $intf] != 1 ||
                    [get_property CONFIG.HAS_TREADY $intf] != 1} {
                    error "Incorrect inferred $name AXI stream interface"
                }
            }
            if {[get_property CONFIG.FREQ_HZ [get_bd_pins vision_dma/aclk]] != 100000000 ||
                [get_property CONFIG.POLARITY [get_bd_pins vision_dma/aresetn]] ne "ACTIVE_LOW"} {
                error "Incorrect inferred DMA clock/reset metadata"
            }
            puts "VISION_DMA_INTERFACE_PASS: S_AXIS/M_AXIS 32-bit, TKEEP/TLAST/TREADY, 100MHz, active-low reset"
        }
        puts $channel {close_bd_design [get_bd_designs dma_interface_check]}
        puts $channel {synth_design -top vision_dma_stream -part xc7z020clg400-1 -mode out_of_context}
        puts $channel {create_clock -name dma_clk -period 10.000 [get_ports aclk]}
        puts $channel {report_utilization -file vision_dma_utilization.rpt}
        puts $channel {report_timing_summary -file vision_dma_timing.rpt}
        puts $channel {puts "VISION_DMA_SYNTH_PASS width=640 height=480"}
        close $channel
        set result [dma_run_tool vivado -mode batch -nojournal -source $synthesis_script \
            -log vision_dma_synth.log]
        if {![string match "*VISION_DMA_INTERFACE_PASS*" $result] ||
            ![string match "*VISION_DMA_SYNTH_PASS*" $result]} {
            error "Missing module-reference inference or synthesis success marker"
        }
    }
} dma_message dma_options]
cd $dma_original_dir
puts "Vision DMA verification elapsed: [expr {([clock milliseconds] - $dma_started) / 1000.0}] seconds"
puts "Vision DMA logs and generated artifacts: $dma_temp"
if {$dma_failed} {
    puts stderr "VISION_DMA_FAIL: $dma_message"
    exit 1
}
puts "VISION_DMA_VERIFICATION_PASS"
