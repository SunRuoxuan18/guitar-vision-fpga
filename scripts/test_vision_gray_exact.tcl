# Standalone: D:/AMD/2026.1/Vivado/bin/xtclsh.bat <this script> [--compile-only]
# Exhaustive RGB888 arithmetic, two-cycle latency, elastic flow and reset checks.
# VIVADO_BIN overrides the tool directory. All generated files stay in TEMP.
set gray_root [file dirname [file dirname [file normalize [info script]]]]
set gray_bin "D:/AMD/2026.1/Vivado/bin"
if {[info exists ::env(VIVADO_BIN)]} { set gray_bin $::env(VIVADO_BIN) }
set gray_started [clock milliseconds]
set gray_deadline [expr {$gray_started + 600000}]
set gray_original_dir [pwd]
set gray_temp [file join $::env(TEMP) "vision_gray_exact_[pid]_[clock clicks]"]
file mkdir $gray_temp
puts "Vision grayscale logs and generated artifacts: $gray_temp"

proc gray_run_tool {name args} {
    global gray_bin gray_deadline
    if {[clock milliseconds] >= $gray_deadline} {
        error "Vision grayscale verification deadline exceeded"
    }
    set executable [file join $gray_bin "${name}.bat"]
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
        if {[clock milliseconds] >= $gray_deadline} {
            foreach child [pid $channel] { catch {exec taskkill /PID $child /T /F 2>@1} }
            catch {close $channel}
            error "Vision grayscale verification exceeded 600 seconds while running $name"
        }
        if {[eof $channel]} { break }
        after 20
    }
    fconfigure $channel -blocking 1
    if {[catch {close $channel} message]} { error "$name failed: $message\n$output" }
    return $output
}

set gray_failed [catch {
    set gray_compile_only 0
    foreach argument $argv {
        if {$argument ne "--compile-only"} { error "Unknown argument: $argument" }
        set gray_compile_only 1
    }
    cd $gray_temp
    gray_run_tool xvlog -sv [file join $gray_root rtl vision_rgb_gray.v] \
        [file join $gray_root sim tb_vision_gray_exact.v] -log vision_gray_exact_compile.log
    gray_run_tool xelab tb_vision_gray_exact -s vision_gray_exact \
        -log vision_gray_exact_elab.log
    if {$gray_compile_only} {
        puts "VISION_GRAY_EXACT_COMPILE_PASS"
    } else {
        set result [gray_run_tool xsim vision_gray_exact -runall -onerror quit \
            -onfinish quit -log vision_gray_exact_sim.log]
        set marker "VISION_GRAY_EXACT_PASS rgb888=16777216 latency=2 throughput=1_pixel_per_clock"
        set lines [split [string map [list "\r" ""] $result] "\n"]
        if {[lsearch -exact $lines $marker] < 0} {
            error "Missing exact successful test result: $marker"
        }
    }
} gray_message gray_options]
cd $gray_original_dir
puts "Vision grayscale verification elapsed: [expr {([clock milliseconds] - $gray_started) / 1000.0}] seconds"
puts "Vision grayscale logs and generated artifacts: $gray_temp"
if {$gray_failed} {
    puts stderr "VISION_GRAY_EXACT_FAIL: $gray_message"
    exit 1
}
if {!$gray_compile_only} { puts "VISION_GRAY_EXACT_VERIFICATION_PASS" }
