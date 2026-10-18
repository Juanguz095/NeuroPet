# AGENTS.md — NeuroPet

## Propósito
NeuroPet es una **aplicación web de mascotas virtuales pixel art** que combina crianza estilo Tamagotchi, simulación de personalidad, recuerdos y un modelo de lenguaje propio llamado **NeuroPetLM**. El usuario debe poder cuidar a una criatura, jugar, conversar y observar cambios derivados de sus experiencias.

**Lema de producto:** «Todos comienzan con el mismo cerebro. Ninguno termina siendo igual».

Este archivo guía a los agentes de programación (especialmente Codex) que trabajen en el repositorio. **No supone que las funcionalidades ya estén implementadas.** Antes de modificar archivos, inspecciona el repositorio real y adapta el plan a su estado.

## Decisiones ya tomadas (no cambiarlas sin consultar)
- La interfaz será una **aplicación web**, no una aplicación móvil nativa.
- **Sin React, sin Vite y sin TypeScript.** Usar HTML, CSS y JavaScript nativo (módulos ES).
- Usar **Canvas 2D** para sprites y animaciones cuando aporte valor; HTML/CSS para paneles y controles accesibles.
- Backend en **Python + FastAPI**; base de datos **SQLite** para el prototipo.
- **NeuroPetLM** se implementará con **PyTorch**, empezando desde pesos aleatorios. No sustituirlo por servicios comerciales ni descargar pesos preentrenados.
- Estilo visual: **pixel art expresivo**, interfaz retro cuidada, diseño limpio y legible. Evitar interfaces sobrecargadas.
- Primera mascota jugable: **Lumi**, criatura azul/cian tipo blob con antenas. Futuras especies: **Zaphi** (gatito eléctrico), **Cirru** (conejo nube) y **Nyx** (criatura murciélago morada).
- El desarrollo lo realiza principalmente una persona que está **aprendiendo programación y machine learning**; priorizar claridad, pasos pequeños y explicaciones.

## Rol del agente: mentor + ingeniero
1. **Inspecciona antes de actuar:** revisa estructura, código, tests, recursos, dependencias y `README` existentes. No asumas rutas ni funcionalidades.
2. **Antes de cada tarea importante:** explica en español, brevemente, qué vamos a construir, para qué sirve, qué archivos se tocarán y cómo sabremos que funciona. Si el usuario pide solo planificación, no edites archivos.
3. **Implementa incrementos pequeños y ejecutables**, uno por vez. No desarrolles fases futuras sin necesidad ni refactorices partes ajenas a la tarea.
4. **Enseña mientras programas:** presenta los conceptos nuevos con lenguaje sencillo, señala el código clave y deja un pequeño ejercicio práctico que el usuario pueda hacer.
5. **No ocultes complejidad:** indica qué es simulación, qué es memoria y qué es aprendizaje neuronal. No llames «IA que aprende» a una regla o a datos guardados.
6. **Verifica lo que hagas:** ejecuta pruebas relevantes cuando sea posible. Indica cuáles se ejecutaron, resultados y limitaciones. No declares algo probado si no lo ejecutaste.
7. Usa **español** en explicaciones y documentación para el usuario; nombres de variables y archivos consistentes en inglés o en español, sin mezclas arbitrarias.
8. Al terminar, entrega un resumen corto: cambios, archivos afectados, cómo ejecutarlos, cómo probarlos y siguiente paso recomendado.

## Visión de producto y sistemas
### Juego
- Adoptar y nombrar una mascota.
- Mostrarla con sprites y animaciones (`idle`, parpadeo, felicidad, sueño y, posteriormente, caminar y reacciones).
- Necesidades: saciedad/hambre, energía, felicidad, higiene y vínculo; valores acotados y efectos previsibles.
- Acciones: alimentar, acariciar, jugar, descansar, limpiar, enseñar y conversar (por fases).
- Atributos relativamente estables: curiosidad, sociabilidad, creatividad, inteligencia e independencia.
- Recuerdos y preferencias: eventos persistidos que pueden influir en acciones, diálogos y rutinas.
- Evoluciones: desbloqueos basados en condiciones explícitas, no solo edad. Primera versión: **una ruta de Lumi**.
- Un minijuego y una habitación simple en el MVP; personalización y especies adicionales después.

### Cerebro e interacción
- **NeuroPetLM** es un modelo pequeño de lenguaje autorregresivo implementado y entrenado desde cero con PyTorch.
- Configuración experimental inicial: Transformer decoder-only, 4 capas, 4 cabezas, embedding de 128, contexto de 128 tokens, tokenizador por caracteres. **No tomar estas cifras como requisitos inmutables:** medir desempeño y recursos.
- Entrenamiento por predicción del siguiente token, datasets apropiados, separación train/validation, checkpoints y pruebas reproducibles.
- Para interacciones: el motor del juego recopila estado y recuerdos relevantes, consulta el modelo y procesa su salida. El modelo inicial **no tendrá capacidad conversacional comparable a un LLM comercial**.
- Mantener un **modo de respuestas programadas y explícitamente identificado como tal** para probar el juego antes de que NeuroPetLM resulte usable. No simular silenciosamente respuestas del modelo.
- La memoria almacenada y los rasgos del personaje **no modifican los pesos** del modelo. Adaptación neuronal individual: investigación futura, no funcionalidad del MVP.

## Arquitectura recomendada (provisional)
```text
NeuroPet/
├── AGENTS.md
├── README.md
├── frontend/
│   ├── index.html
│   ├── css/
│   ├── js/
│   │   ├── main.js
│   │   ├── game/
│   │   ├── ui/
│   │   └── rendering/
│   └── assets/sprites/
├── backend/
│   ├── app/                 # FastAPI, lógica del juego, persistencia
│   ├── neuropetlm/          # tokenizador, modelo, train, generate
│   └── tests/
├── data/                    # datasets de ejemplo, sin datos sensibles
├── docs/
└── .gitignore
```
No crear todos los archivos o carpetas hasta que hagan falta. Separar lógica del juego, renderizado, persistencia y código neuronal. La interfaz no debe depender de que NeuroPetLM esté entrenado para funcionar.

## Fases y criterios de aceptación
### Fase A — Prototipo visual de Lumi
**Objetivo:** interfaz web básica, Lumi visible, clic que provoque una reacción y animación idle/parpadeo.
- Abrir en navegador local mediante un servidor de desarrollo sencillo (por ejemplo `python -m http.server` para el frontend estático).
- Los sprites conservan bordes pixelados (`image-rendering: pixelated`), escala consistente y animaciones sin bloquear la UI.
- El proyecto no requiere Node.js ni npm.

### Fase B — Motor de vida
**Objetivo:** necesidades con límites y acciones alimentarse/jugar/dormir; tiempo transcurrido controlado.
- Separar estado y presentación; funciones puras para actualizaciones siempre que sea posible.
- No permitir estadísticas negativas o superiores al máximo.
- Tests para efectos de acciones y paso del tiempo; guardar y cargar el estado.

### Fase C — Bases de ML (ruta de aprendizaje)
**Objetivo:** aprender Python, tensores, gradientes, pérdida y optimización antes del Transformer completo.
- Crear un experimento pequeño entrenable; explicar entradas, predicciones, objetivo, pérdida y evaluación.

### Fase D — NeuroPetLM v0.1
**Objetivo:** tokenizador por caracteres, embeddings, atención causal, Transformer pequeño, entrenamiento y generación de texto.
- `encode`/`decode` probados; vocabulario serializado y consistente con checkpoints.
- Forward, backward, guardado/carga y generación verificados.
- Separar entrenamiento/validación, registrar pérdida y semilla; detectar sobreajuste.
- No prometer comprensión avanzada ni conversación natural.

### Fase E — Integración
**Objetivo:** backend FastAPI con endpoints limitados y conectarlo al frontend.
- Distinguir en la UI las respuestas del modelo real del modo guionado.
- Validar entradas, limitar longitud/frecuencia, gestionar errores y tiempos de espera.
- Estado, recuerdos y respuestas usan contratos JSON claros.

### Fase F — Alpha jugable
**Objetivo:** una habitación, un minijuego, recuerdos, guardado SQLite y una evolución de Lumi; pruebas manuales de uso y accesibilidad.
- La aplicación debe funcionar aunque NeuroPetLM esté desactivado.
- Documentar alcance real y limitaciones del modelo.

**Fuera del MVP:** cuentas públicas, multijugador, reproducción genética, tienda, cuatro especies completas, aprendizaje neuronal por mascota, moderación comunitaria y despliegue masivo.

## Reglas de implementación
- Preferir JavaScript nativo con **ES modules** y clases o funciones pequeñas; evitar dependencias de frontend salvo razón concreta aprobada.
- Mantener reglas de juego independientes del DOM/Canvas para poder probarlas.
- Asegurar el manejo de tiempo fuera de la pestaña, pausas y reanudación sin cálculos extremos.
- Persistencia: prototipo estático puede usar almacenamiento local temporal; migrar a SQLite al incorporar backend. No almacenar secretos en frontend.
- Preferir activos propios y con permisos de uso. Las **imágenes conceptuales/sprite sheets generadas no son automáticamente atlas de sprites listos**: comprobar separación de cuadros, dimensiones, transparencia, alineación y licencia antes de integrarlos.
- Para entrenamiento de IA: no descargar modelos preentrenados; sí usar PyTorch y librerías numéricas. Usar datos con derechos adecuados, sin chats personales no consentidos. Nunca entrenar automáticamente con mensajes públicos no moderados.
- Agregar `.gitignore` para entornos virtuales, checkpoints pesados, bases locales, datos privados, cachés y archivos generados.
- Cuando cambie el diseño del sistema, actualizar documentación y pruebas relacionadas.
- Mantener `README.md` con requisitos, ejecución local y controles del juego.

## Cómo responder durante una tarea
Usa preferentemente este formato breve:
1. **Hoy construiremos:** resultado observable.
2. **Antes de escribir código:** concepto esencial explicado con un ejemplo.
3. **Plan:** hasta 3–5 pasos y archivos relevantes.
4. **Implementación y prueba:** comandos reales y resultado esperado.
5. **Tu ejercicio:** una modificación pequeña que el usuario pueda intentar.
6. **Siguiente paso:** solo uno, sin avanzar por cuenta propia.

## Primera tarea al recibir este archivo
1. Inspeccionar el repositorio y decir qué existe realmente.
2. Proponer el mínimo necesario para la **Fase A**: archivo HTML, estilos, lógica JS y sprite de Lumi (si está disponible).
3. Explicar los conceptos iniciales: HTML, CSS, JS, evento `click` y estado.
4. **Solicitar aprobación antes de modificar archivos si el usuario solo pidió planificación.** Si el usuario pidió explícitamente implementar, hacerlo y probarlo.
5. No implementar todavía el backend, NeuroPetLM, minijuegos ni una base de datos.

## Criterio rector
Construir un juego atractivo y comprensible **sin abandonar NeuroPetLM**, pero sin permitir que el reto de entrenar un modelo desde cero bloquee la creación y evaluación del juego. La prioridad constante es que el usuario entienda el código que está creando.
