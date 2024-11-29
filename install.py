import os
import platform
import subprocess
import sys

def run_command(command, shell=False):
    """Exécute une commande système et affiche les résultats."""
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erreur : La commande '{' '.join(command)}' a échoué.")
        sys.exit(1)

def install_tp():
    """Automatise l'installation pour Linux et Windows."""
    python_exe = sys.executable  # Utilise le même interpréteur Python
    is_windows = platform.system() == "Windows"

    # Étape 1 : Création d'un environnement virtuel
    venv_dir = ".venv"
    run_command([python_exe, "-m", "venv", venv_dir])

    # Étape 2 : Activer l'environnement virtuel et installer les dépendances
    pip_cmd = [
        os.path.join(venv_dir, "Scripts" if is_windows else "bin", "pip")
    ]

    # Installer les bibliothèques nécessaires
    dependencies = ["./bsplyne", "pyxel-dic", "numba==0.56.4", "ipykernel"]
    run_command(pip_cmd + ["install"] + dependencies)

    # Étape 3 : Ajouter l'environnement comme kernel Jupyter
    kernel_cmd = [
        os.path.join(venv_dir, "Scripts" if is_windows else "bin", "python"),
        "-m", "ipykernel", "install", "--user", f"--name={venv_dir}", "--display-name", "Python (TP VIC venv)"
    ]
    run_command(kernel_cmd)

    print("\nInstallation terminée avec succès !")

if __name__ == "__main__":
    install_tp()