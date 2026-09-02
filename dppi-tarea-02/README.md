# Más gatitos, la venganza

## tarea-02

- **Integrante-1** Victoria Sol Frias
- **Integrante-2** Débora Soto

- Asignatura: Dispositivos Periféricos y Plataformas para la Interacción Digital **DIS9087**

Proyecto de reconocimiento de gestos, utilizando Python y MediaPipe. Realizado tomando como referencia este repositorio:

- <https://github.com/catherpiee/meowmeowcatcam>

## Gestos

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
| - | `default` | Estado neutro | En reposo / sin ningún gesto activo | `7.Duda.jpeg` |

Las imágenes de los memes se encuentran en la carpeta de `Gatitoactualizado/`).

## Ejecución — Escritorio (Python)

Requiere Python 3 y cámara web.

La forma más sencilla: doble clic en **`Iniciar Servidor Web.command`**.

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
Serving HTTP on :: port 8080 (http://[::]:8080/)
```

Luego abre en tu navegador `http://localhost:8080/` y autoriza el acceso a la cámara.

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
 [carpeta de imágenes](./Gatitoactualizado)

- [video](./Video)
