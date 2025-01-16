import os

class DirectoryTree:
    """
    Classe pour générer l'architecture d'un dossier et l'enregistrer dans un fichier texte.
    """

    def __init__(self, root_dir: str, output_file: str):
        """
        Initialise l'objet avec le chemin du dossier racine et le fichier de sortie.

        :param root_dir: Chemin du dossier racine.
        :param output_file: Chemin du fichier où sera enregistrée l'architecture.
        """
        self.root_dir = root_dir
        self.output_file = output_file

    def generate_tree(self) -> None:
        """
        Génère l'architecture du dossier et l'enregistre dans un fichier texte.
        """
        with open(self.output_file, "w") as file:
            for root, dirs, files in os.walk(self.root_dir):
                level = root.replace(self.root_dir, "").count(os.sep)
                indent = " " * 4 * level
                file.write(f"{indent}{os.path.basename(root)}/\n")
                sub_indent = " " * 4 * (level + 1)
                for f in files:
                    file.write(f"{sub_indent}{f}\n")

if __name__ == "__main__":
    root_directory = "./"  # Remplacez par le chemin du dossier cible
    output_file = "directory_structure.txt"
    tree = DirectoryTree(root_dir=root_directory, output_file=output_file)
    tree.generate_tree()
