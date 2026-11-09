# Reset only FCLK0's PL logic; the PS, DDR and peripherals remain running.
connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "APU"}
set previous_reset [lindex [mrd -value 0xF8000240] 0]
set slcr_locked [expr {[lindex [mrd -value 0xF800000C] 0] & 1}]
try {
    if {$slcr_locked} { mwr 0xF8000008 0x0000DF0D }
    mwr 0xF8000240 [expr {$previous_reset | 1}]
    after 20
    mwr 0xF8000240 [expr {$previous_reset & 0xFFFFFFFE}]
    puts "AUDIO_AXI_RESET: [mrd 0xF8000240]"
} finally {
    if {$slcr_locked} { mwr 0xF8000004 0x0000767B }
}
disconnect
