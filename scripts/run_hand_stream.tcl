set root [file dirname [file dirname [file normalize [info script]]]]
# Supply the same absolute path to Vivado's -log argument.
set ::hand_stream_native_log "$root/diagnostics/hand_stream_native.log"
source -notrace "$root/scripts/build_hand_stream.tcl"
source -notrace "$root/scripts/implement_hand_stream.tcl"
