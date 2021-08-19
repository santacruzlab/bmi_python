#define REWARD1 22
#define REWARD2 23

int led = 13;
int strobe  = 48;
int rstart = 49;
int di0 = 36;
char c;
char d;
int en = 0;
char dio_data[2];
int data_pins[] = {26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 40, 41, 42, 43, 46, 47};

int rstart_init = 0;

void setup() {
  pinMode(REWARD1, OUTPUT);
  pinMode(REWARD2, OUTPUT);
  
  pinMode(strobe, OUTPUT) ;  
  
  // initialize the output pins
  for (int k = 0; k < (sizeof(data_pins)/sizeof(int)); k += 1) {
    pinMode(data_pins[k], OUTPUT);
    digitalWrite(data_pins[k], LOW);
  }
  pinMode(led, OUTPUT);
  
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() >= 1) {
    c = Serial.read();
    
    if (c == 'j') {
      digitalWrite(REWARD1, HIGH);
    }
    else if (c == 'r') {
      digitalWrite(REWARD2, HIGH);
    }
    else if (c == 'n') {
      digitalWrite(REWARD1, LOW);
    }
    else if (c == 'o') {
      digitalWrite(REWARD2, LOW);
    }

    // Start recording
    else if ((c == 'u') && (en == 0)) {
        if (rstart_init == 0) {
          pinMode(rstart, OUTPUT);        
          rstart_init = 1;
        }
        
        // positive edge for rstart
        digitalWrite(rstart, HIGH);
        //delay(200);
        //digitalWrite(rstart, LOW);
  
        en = 1;    
    }
    
    // Stop recording
    else if ((c == 'p') && (en == 1)) {
        digitalWrite(rstart, LOW);
        en = 0;
        c = ' ';

        for (int k = 0; k < (sizeof(data_pins)/sizeof(int)); k += 1) {
          digitalWrite(data_pins[k], LOW);
        }
        
        // turn on LED (debugging)      
        digitalWrite(led, HIGH);
        //delay(500);
        //digitalWrite(led, LOW);
    }
    
    // Digital data
    else if (c == 'd') {
      handle_word();
    }  
    
  }
  
}

void handle_word() { 
  Serial.readBytes(dio_data, 2);
  char d1 = dio_data[0];
  char d0 = dio_data[1];
  
  
    // set all the data bits
    for (int byte_idx = 0; byte_idx < 2; byte_idx += 1) {
      byte data_byte = dio_data[byte_idx];
      for (int bit_idx = 0; bit_idx < 8; bit_idx += 1) {
        int pin_idx = 8*byte_idx + bit_idx;
        byte mask = 1 << bit_idx;
        if (mask & data_byte) {
          digitalWrite(data_pins[pin_idx], HIGH);
        } else {
          digitalWrite(data_pins[pin_idx], LOW);         
        }
      }
    }  

//  if (d0 & 00000001) 
//    digitalWrite(di0, LOW);
//  else
//    digitalWrite(di0, HIGH);      

  digitalWrite(strobe, HIGH);
  delay(0.5);
  digitalWrite(strobe, LOW);  
}
