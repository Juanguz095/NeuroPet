# NeuroPetLM v0.1 MVP

Este prototipo entrena desde cero un modelo de caracteres pequeño y genera texto desde una terminal. No usa pesos preentrenados ni ofrece aún memoria individual, personalidad persistente o una interfaz web.

## Preparar el entorno

Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r neuropetlm/requirements.txt
```

## Entrenar

```bash
python -m neuropetlm.src.train
```

El entrenamiento usa `neuropetlm/data/conversaciones.txt`. Los pesos empiezan aleatoriamente; el script crea un vocabulario con la parte de entrenamiento y guarda el mejor checkpoint y `tokenizer.json` en `neuropetlm/checkpoints/`. La pérdida de entrenamiento y validación se imprime durante el proceso.

## Generar texto

```bash
python -m neuropetlm.src.generate --prompt "Usuario: Hola. NeuroPet:"
```

Opciones: `--max-new-tokens`, `--temperature` y `--top-k`.

## Ejecutar las pruebas

```bash
python -m unittest discover -s neuropetlm/tests -v
```

Las pruebas de tokenizer funcionan con la biblioteca estándar. Las pruebas de embeddings, atención y modelo necesitan PyTorch.

## Alcance del prototipo

El corpus de ejemplo es pequeño y sirve para comprobar el flujo de datos, entrenamiento, checkpoint y generación. No se espera una conversación fiable ni conocimiento general. Para evaluar generalización se necesitará después un corpus mayor y una validación más amplia.
