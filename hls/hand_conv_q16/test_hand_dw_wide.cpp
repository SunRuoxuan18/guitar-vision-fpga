#define main full_reference_main
#include "test_hand_conv_q16.cpp"
#undef main

int main() {
    bool ok = full_reference_main() == 0;
    ok &= check(5,35,16,64,1,1,0,0,0,0,1);
    ok &= check(5,33,24,17,1,2,1,1,0,0,1);
    ok &= check(7,35,11,11,3,1,1,1,1,1,11);
    ok &= check(9,37,9,9,5,2,2,1,1,2,9);
    ok &= check(3,3,1,1,5,1,2,2,2,2,1);
    return ok ? 0 : 1;
}
