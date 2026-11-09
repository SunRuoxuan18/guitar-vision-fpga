connect -url tcp:127.0.0.1:3121
foreach core {0 1} {
    targets -set -filter "name =~ \"ARM Cortex-A9 MPCore #$core\""
    stop
    puts "CORE$core: [rrd pc]"
    con
}
disconnect
