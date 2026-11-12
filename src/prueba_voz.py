import os
import env_loader  # carga claves desde .env
import pyttsx3
import requests

api_key = os.environ.get("OPENAI_API_KEY", "")
gpt_url = "https://api.openai.com/v1/chat/completions"

print("🎙️ Iniciando prueba de voz y API...")

voz = pyttsx3.init()
voz.setProperty('rate', 150)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Di exactamente esta frase: 'Sistemas en línea. ATOM está listo.'"}],
    "max_tokens": 50
}

# --- AQUÍ EMPIEZA LA MAGIA ---
res = requests.post(gpt_url, json=payload, headers=headers)
res_json = res.json()

print("\n📦 Paquete JSON recibido con éxito.")

# MIRA ESTA LÍNEA CON LUPA:
# Entramos a 'choices', LUEGO al elemento, LUEGO a 'message', LUEGO a 'content'
mensaje = res_json['choices'][0]['message']['content']

print(f"\n✅ Mensaje extraído correctamente: {mensaje}")
print("🗣️ Hablando...")

voz.say(mensaje)
voz.runAndWait()
print("🎉 Prueba finalizada con éxito.")