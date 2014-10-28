#include "mbed.h"
#include "FastAnalogIn.h"
#include "EthernetInterface.h"
 
AnalogIn input1(p15);
AnalogIn input2(p16);
AnalogIn input3(p17);
AnalogIn input4(p18);
DigitalOut led1(LED1);
Timer t, t2;

const int BROADCAST_PORT = 58083;
 
int main() {
    //Setting up the Ethernet
    EthernetInterface eth;
    eth.init(); //Use DHCP
    eth.connect();
    
    UDPSocket sock;
    sock.init();
    sock.set_broadcasting();
    
    Endpoint broadcast;
    broadcast.set_address("255.255.255.255", BROADCAST_PORT);
    
    uint16_t sample_buffer[2561];
    sample_buffer[2560] = 0;

    while(1){   
        printf("Executing Read Loop");
        t.start();
        for(int i=0; i<512; i++) {
            sample_buffer[i] = (uint16_t)t.read_us();
            sample_buffer[i+512] = input1.read_u16();
            sample_buffer[i+1024] = input2.read_u16();
            sample_buffer[i+1536] = input3.read_u16();
            sample_buffer[i+2048] = input4.read_u16();
            //wait_ms(1);
        }
        t.stop();
        t.reset();
        t2.start();
        
        sample_buffer[2560] = (uint16_t)t2.read_us();
        
        printf("Copying to char array \n");
        char out_buffer[5122];
        memcpy(&out_buffer,&sample_buffer,sizeof(sample_buffer));
        
        printf("Sending to Ethernet \n");
        sock.sendTo(broadcast, out_buffer, sizeof(out_buffer));
        
        t2.stop();
        t2.reset();
    }
}
