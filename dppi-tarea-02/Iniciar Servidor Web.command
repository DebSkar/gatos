#!/bin/bash
cd "$(dirname "$0")"
echo "======================================================="
echo "🐱 Iniciando Gatito Detector - 8 Gestos (Versión Web)"
echo "======================================================="
echo "Carpeta: $(pwd)"
echo "Abriendo navegador en: http://localhost:8080"
echo "Para cerrar el servidor, presiona Ctrl + C o cierra esta ventana."
echo "======================================================="

# Abrir el navegador en el puerto 8080
sleep 1 && open "http://localhost:8080" &

# Iniciar servidor HTTP en el puerto 8080
python3 -m http.server 8080
