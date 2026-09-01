# Meowmeow Cat Cam Meme Detector

Apunta tu cámara web hacia ti, haz un gesto con las manos o el rostro y obtén un meme de gatito en tiempo real. Funciona tanto como aplicación de escritorio (ventanas OpenCV en Python) como en el navegador (MediaPipe WASM sin necesidad de instalación).

Muestra dos vistas lado a lado:
- **Cámara** — Tu video en vivo con los landmarks dibujados y un HUD de depuración con métricas (Gesto, Pitch, Apertura de boca, Yaw).
- **Meme** — El meme del gatito correspondiente al gesto que estás realizando.

## Gestos Soportados (`Gestos.txt`)

Se evalúan en el siguiente orden de prioridad:

| # | Identificador | Gesto | Cómo activarlo | Meme asociado |
|---|---|---|---|---|
| 1 | `orejaGato` | Orejas de gato con los dedos | Dos manos arriba de la cabeza, dedos índice y medio estirados hacia abajo | `1.Oreja de gato.jpeg` |
| 2 | `brazosAtras` | Brazos hacia atrás | Levantar brazos y echarlos hacia los lados / atrás | `5.Brazos hacia atras.jpeg` |
| 3 | `shocked` | Brazos cruzados | Cruzar brazos a la altura del pecho | `6.Shocked.jpeg` |
| 4 | `duda` | Cara de duda | Dedo índice estirado tocando el lateral de la cabeza / sien | `7.Duda.jpeg` |
| 5 | `pistola` | Dedos en pistola | Dedos índice y medio juntos hacia arriba, pulgar hacia el lado | `8.Pistola.jpeg` |
| 6 | `lengua` | Sacar la lengua | Abrir la boca / sacar la lengua | `3.Lengua.jpeg` |
| 7 | `cabezaAtras` | Cabeza hacia atrás | Inclinar la cabeza hacia atrás / mirar hacia arriba (Pitch $> +13^\circ$) | `4.CabezaHaciaAtras.jpeg` |
| 8 | `mirarAbajo` | Cabeza hacia abajo | Inclinar la cabeza hacia abajo / mirar hacia el suelo (Pitch $< -13^\circ$) | `2.MirarHaciaAbajo.jpeg` |
| - | `default` | Estado neutro | En reposo / sin ningún gesto activo | `pokercat.jpg` |

Las imágenes de los memes se encuentran en la carpeta `memes/` (provenientes de `Gatitoactualizado/`).

## Ejecución — Escritorio (Python)

Requiere Python 3 y cámara web.

La forma más sencilla: doble clic en **`Launch Gesture Meme.command`**.

O de forma manual desde la terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 gesture_meme.py
```

Presiona `q` o `Esc` en la ventana de la cámara para salir.

## Ejecución — Navegador Web

No requiere instalación, solo servir los archivos por HTTP (para los permisos de la webcam):

```bash
python3 -m http.server 8000
```

Luego abre en tu navegador `http://localhost:8000` y autoriza el acceso a la cámara.

## HUD de Depuración en Vivo

La ventana de la cámara muestra un texto en la esquina superior con las métricas en tiempo real:

```
Gesto: cabezaAtras
Pitch: +18.4 deg  (atras >+13, abajo <-13)
Mouth Open: 0.08  (lengua >0.24)
Yaw: -2.1 deg
```

## Estructura del Proyecto

```
gesture_meme.py       Versión de escritorio (OpenCV + MediaPipe Tasks Python)
app.js                Versión web (MediaPipe tasks-vision WASM)
index.html            Interfaz visual web
memes/                Imágenes de los memes de gatitos
models/               Modelos .task de MediaPipe para la versión de escritorio
Gestos.txt            Definición de gestos requeridos
Gatitoactualizado/    Carpeta fuente con las imágenes originales de los memes
requirements.txt      Dependencias de Python (opencv-python, mediapipe, numpy)
```

