# DeepLearningExe

## Voraussetzungen

* Python 3.7 oder höher
* Pip (Python-Paketmanager)
* Virtuelle Umgebung (empfohlen)

## Installation

1. **Repository klonen**

   ```bash
   git clone https://github.com/DeinBenutzername/DeepLearningExe.git
   cd DeepLearningExe
   ```

2. **Virtuelle Umgebung erstellen (optional, aber empfohlen)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Abhängigkeiten installieren**

   Der Ordner enthält eine Datei `requirements.txt`, in der alle benötigten Python-Pakete und deren Versionen festgehalten sind. Um diese Pakete zu installieren, führst du folgenden Befehl aus:

   ```bash
   pip install -r requirements.txt
   ```

   * Der Parameter `-r` steht für „requirements“ und weist Pip an, die aufgelisteten Pakete aus der Datei zu installieren.
   * Solltest du bereits eine ältere Version eines Pakets installiert haben, wird Pip es auf die in `requirements.txt` angegebene Version aktualisieren.

## Nutzung

Nach der erfolgreichen Installation kannst du die Skripte im Projekt ausführen. Beispielsweise:

```bash
python train_model.py --config config/train_config.yaml
```

Passe dabei die Pfade zu Konfigurationsdateien oder Datensätzen nach Bedarf an.

## Aktualisierung der `requirements.txt`

Solltest du während der Weiterentwicklung neue Pakete installieren und diese in der Requirements-Datei festhalten wollen, kannst du:

```bash
pip freeze > requirements.txt
```

Dadurch wird deine aktuelle Umgebung ausgelesen und alle Abhängigkeiten in `requirements.txt` geschrieben.

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe die Datei [LICENSE](LICENSE) für Details.
