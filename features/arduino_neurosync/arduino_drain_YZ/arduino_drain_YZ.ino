#define REWARD1 22
#define REWARD2 23

void setup() {
  pinMode(REWARD1, OUTPUT);
  pinMode(REWARD2, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  digitalWrite(REWARD1, HIGH);
  digitalWrite(REWARD2, HIGH);
}
