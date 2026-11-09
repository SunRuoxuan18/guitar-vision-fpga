# Standalone: D:/AMD/2026.1/Vivado/bin/xtclsh.bat <this script> [--synth]
# Or source from Vivado batch Tcl. No project is opened or modified.
set vision_root [file dirname [file dirname [file normalize [info script]]]]
set vision_bin "D:/AMD/2026.1/Vivado/bin"
if {[info exists ::env(VIVADO_BIN)]} { set vision_bin $::env(VIVADO_BIN) }
set vision_started [clock milliseconds]
set vision_deadline [expr {$vision_started + 115000}]
set vision_original_dir [pwd]
set vision_temp [file join $::env(TEMP) "vision_preprocess_[pid]_[clock seconds]"]
file mkdir $vision_temp

proc vision_run_tool {name args} {
    global vision_bin vision_deadline
    if {[clock milliseconds] >= $vision_deadline} { error "Vision test wall-time budget exhausted" }
    set executable [file join $vision_bin "${name}.bat"]
    if {![file exists $executable]} { error "Missing Vivado tool: $executable" }
    set command [list $::env(COMSPEC) /c $executable {*}$args]
    puts "Running $name"
    set channel [open [concat [list |] $command [list 2>@1]] r]
    fconfigure $channel -blocking 0 -buffering none
    set output ""
    while {1} {
        set chunk [read $channel]
        append output $chunk
        puts -nonewline $chunk
        flush stdout
        if {[eof $channel]} { break }
        if {[clock milliseconds] >= $vision_deadline} {
            foreach child [pid $channel] {
                catch {exec taskkill /PID $child /T /F 2>@1}
            }
            catch {close $channel}
            error "Vision test exceeded 115 seconds while running $name"
        }
        after 20
    }
    fconfigure $channel -blocking 1
    if {[catch {close $channel} message]} { error "$name failed: $message\n$output" }
    return $output
}

set vision_failed [catch {
    cd $vision_temp
    vision_run_tool xvlog -sv [file join $vision_root rtl vision_rgb_gray.v] \
        [file join $vision_root rtl vision_sobel3x3.v] \
        [file join $vision_root sim tb_vision_preprocess.v]
    foreach width {5 3} {
        set snapshot "vision_preprocess_w${width}"
        set top tb_vision_preprocess
        if {$width == 3} { set top tb_vision_preprocess_min3 }
        vision_run_tool xelab $top -s $snapshot -debug typical -log "${snapshot}_elab.log"
        set result [vision_run_tool xsim $snapshot -runall -onerror quit -onfinish quit \
            -log "${snapshot}_sim.log"]
        if {![string match "*VISION_PREPROCESS_PASS width=$width *" $result]} {
            error "Missing successful TB result for FRAME_WIDTH=$width"
        }
    }
    puts "VISION_SIMULATION_PASS: byte-exact pipeline, FRAME_WIDTH=5 and minimum FRAME_WIDTH=3"

    if {[info exists argv] && [lsearch -exact $argv --synth] >= 0} {
        set synthesis_script [file join $vision_temp synthesis.tcl]
        set channel [open $synthesis_script w]
        puts $channel [list read_verilog [file join $vision_root rtl vision_rgb_gray.v]]
        puts $channel [list read_verilog [file join $vision_root rtl vision_sobel3x3.v]]
        puts $channel {synth_design -top vision_sobel3x3 -part xc7z020clg400-1 -mode out_of_context}
        puts $channel {report_utilization}
        puts $channel {puts "VISION_SOBEL_SYNTH_PASS width=640"}
        puts $channel {close_design}
        puts $channel [list read_verilog [file join $vision_root rtl vision_rgb_gray.v]]
        puts $channel {synth_design -top vision_rgb_gray -part xc7z020clg400-1 -mode out_of_context}
        puts $channel {report_utilization}
        puts $channel {puts "VISION_GRAY_SYNTH_PASS"}
        close $channel
        set result [vision_run_tool vivado -mode batch -nojournal -nolog -source $synthesis_script]
        if {![string match "*VISION_SOBEL_SYNTH_PASS*" $result] ||
            ![string match "*VISION_GRAY_SYNTH_PASS*" $result]} {
            error "Missing successful standalone synthesis results"
        }
    }
} vision_message vision_options]
cd $vision_original_dir
puts "Vision verification elapsed: [expr {([clock milliseconds] - $vision_started) / 1000.0}] seconds"
puts "Vision-only generated artifacts: $vision_temp"
if {$vision_failed} {
    puts stderr "VISION_PREPROCESS_FAIL: $vision_message"
    exit 1
}
