import os
import platform
import subprocess
import sys
import urllib.request
import shutil
import time

def run_command(command, shell=False):
    """Exécute une commande système et affiche les erreurs en cas d'échec."""
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erreur : La commande '{' '.join(command)}' a échoué.")
        sys.exit(1)

def download_python_installer(url, installer_path):
    """Télécharge l'installateur Python depuis une URL."""
    print("Téléchargement de Python 3.9...")
    urllib.request.urlretrieve(url, installer_path)
    print("Téléchargement terminé.")

def install_python_windows():
    """Télécharge et installe Python 3.9 sur Windows."""
    python_installer_url = (
        "https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe"
    )
    installer_path = "python-3.9.13-amd64.exe"

    if not os.path.exists(installer_path):
        download_python_installer(python_installer_url, installer_path)

    print("Installation de Python 3.9...")
    run_command([installer_path, "/quiet", "InstallAllUsers=1", "PrependPath=1"])
    os.remove(installer_path)  # Nettoyage après installation
    print("Python 3.9 installé avec succès.")

def install_python_linux():
    """Installe Python 3.9 sur Linux via apt."""
    print("Installation de Python 3.9 via apt...")
    run_command(["sudo", "apt-get", "update"])
    run_command(["sudo", "apt-get", "install", "-y", "python3.9", "python3.9-venv"])
    print("Python 3.9 installé avec succès.")

def verify_python_version():
    """Vérifie que Python 3.9 est installé, sinon l'installe."""
    print("Vérification de la version de Python...")
    try:
        output = subprocess.check_output(["python3.9", "--version"], text=True)
        print(f"Python trouvé : {output.strip()}")
    except FileNotFoundError:
        if platform.system() == "Windows":
            install_python_windows()
            # Attendre que Python soit accessible dans le PATH
            time.sleep(5)  # Laisser un peu de temps pour que le PATH soit pris en compte
            print("Vérification après installation...")
            try:
                output = subprocess.check_output(["python3.9", "--version"], text=True)
                print(f"Python trouvé après installation : {output.strip()}")
            except FileNotFoundError:
                print("Erreur : Python 3.9 introuvable après l'installation.")
                sys.exit(1)
        elif platform.system() == "Linux":
            install_python_linux()
        else:
            print("Système non supporté pour l'installation automatique de Python.")
            sys.exit(1)

def main():
    """Automatise l'installation de Python et du TP."""
    verify_python_version()

    # Utiliser Python 3.9 pour créer l'environnement virtuel
    python_exe = shutil.which("python3.9")
    if not python_exe:
        print("Erreur : Python 3.9 introuvable après l'installation.")
        sys.exit(1)

    print(f"Utilisation de Python : {python_exe}")
    venv_dir = ".venv"
    run_command([python_exe, "-m", "venv", venv_dir])

    pip_cmd = [
        os.path.join(venv_dir, "Scripts", "pip") if platform.system() == "Windows" else os.path.join(venv_dir, "bin", "pip")
    ]

    # Installer les dépendances
    dependencies = ["./bsplyne", "pyxel-dic", "numba==0.56.4", "ipykernel"]
    run_command(pip_cmd + ["install"] + dependencies)

    kernel_cmd = [
        os.path.join(venv_dir, "Scripts", "python") if platform.system() == "Windows" else os.path.join(venv_dir, "bin", "python"),
        "-m", "ipykernel", "install", "--user", f"--name={venv_dir}", "--display-name", "Python (TP VIC venv)"
    ]
    run_command(kernel_cmd)
    print("Installation terminée avec succès !")

if __name__ == "__main__":
    main()
