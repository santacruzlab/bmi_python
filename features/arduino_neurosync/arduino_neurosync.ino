#define REWARD1 22
#define REWARD2 23

void setup() {
  pinMode(REWARD1, OUTPUT);
  pinMode(REWARD2, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    char serialListener = Serial.read();
    if (serialListener == 'j') {
      digitalWrite(REWARD1, HIGH);
    }
    else if (serialListener == 'r') {
      digitalWrite(REWARD2, HIGH);
    }
    else if (serialListener == 'n') {
      digitalWrite(REWARD1, LOW);
      digitalWrite(REWARD2, LOW);
    }
  }
  
}
