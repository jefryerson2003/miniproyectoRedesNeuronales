// Pines de entrada para los potenciómetros
const int potPin1 = A0;
const int potPin2 = A1;
const int potPin3 = A2;

// Pin de salida para el LED
const int ledPin = 9;

// Pesos y sesgos simulados para la red neuronal
float weightsInputToHidden[3] = {0.5, -0.3, 0.8}; // Pesos de la capa de entrada a la capa oculta
float biasHidden = 0.2; // Sesgo de la capa oculta
float weightHiddenToOutput = 1.2; // Peso de la capa oculta a la salida
float biasOutput = -0.5; // Sesgo de la salida

void setup() {
  // Configura el pin del LED como salida
  pinMode(ledPin, OUTPUT);
  // Configura los pines de los potenciómetros como entrada
  pinMode(potPin1, INPUT);
  pinMode(potPin2, INPUT);
  pinMode(potPin3, INPUT);

  Serial.begin(9600); // Inicializa el puerto serial para depuración
}

void loop() {
  // Lee los valores de los potenciómetros (0-1023)
  int sensorValue1 = analogRead(potPin1);
  int sensorValue2 = analogRead(potPin2);
  int sensorValue3 = analogRead(potPin3);

  // Normaliza los valores de los sensores a 0-1
  float input1 = sensorValue1 / 1023.0;
  float input2 = sensorValue2 / 1023.0;
  float input3 = sensorValue3 / 1023.0;

  // Calcula la salida de la capa oculta (simulación de activación ReLU)
  float hiddenOutput = max(0.0, input1 * weightsInputToHidden[0] +
                                  input2 * weightsInputToHidden[1] +
                                  input3 * weightsInputToHidden[2] +
                                  biasHidden);

  // Calcula la salida final (simulación de función de activación lineal)
  float output = hiddenOutput * weightHiddenToOutput + biasOutput;

  // Mapea la salida final al rango 0-255 para controlar el brillo del LED
  int ledBrightness = map(output * 100, 0, 100, 0, 255);

  // Asegura que la salida esté dentro del rango permitido
  ledBrightness = constrain(ledBrightness, 0, 255);

  // Ajusta el brillo del LED basado en la salida de la red
  analogWrite(ledPin, ledBrightness);

  // Imprime los resultados en el monitor serial para depuración
  Serial.print("Inputs: ");
  Serial.print(input1);
  Serial.print(", ");
  Serial.print(input2);
  Serial.print(", ");
  Serial.print(input3);
  Serial.print(" | Output: ");
  Serial.print(output);
  Serial.print(" | LED Brightness: ");
  Serial.println(ledBrightness);

  delay(100); // Espera 100 ms antes de la siguiente iteración
}