import os
import re
import h5py
import numpy as np

def extraire_plage_canaux(chemin_txt):
    """Lit le fichier .txt pour trouver la plage des canaux actifs."""
    with open(chemin_txt, 'r', encoding='utf-8') as f:
        contenu = f.read()
    
    # Recherche l'expression "Plage des canaux actifs : de X à Y"
    match = re.search(r"Plage des canaux actifs\s*:\s*de\s+(\d+)\s+à\s+(\d+)", contenu)
    if match:
        canal_debut = int(match.group(1))
        canal_fin = int(match.group(2))
        return canal_debut, canal_fin
    else:
        raise ValueError("Impossible de trouver la ligne 'Plage des canaux actifs : de X à Y' dans le fichier texte.")

def copier_canaux_actifs(chemin_h5_orig, chemin_txt, dossier_sortie):
    # 1. Extraire la plage des canaux du fichier texte
    canal_debut, canal_fin = extraire_plage_canaux(chemin_txt)
    print(f"Plage de canaux détectée : de {canal_debut} à {canal_fin}")
    
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(dossier_sortie, exist_ok=True)
    nom_fichier_h5 = os.path.basename(chemin_h5_orig)
    chemin_h5_sortie = os.path.join(dossier_sortie, f"filtre_{nom_fichier_h5}")
    
    # Le chemin exact du dataset brut dans la structure HDF5 du papier est :
    # /Acquisition/Raw[0]/RawData
    chemin_dataset_brut = "/Acquisition/Raw[0]/RawData"
    
    # 2. Ouvrir le fichier original en lecture et le nouveau en écriture
    with h5py.File(chemin_h5_orig, 'r') as f_source, h5py.File(chemin_h5_sortie, 'w') as f_dest:
        
        # Vérification de l'existence du dataset
        if chemin_dataset_brut not in f_source:
            raise KeyError(f"Le dataset '{chemin_dataset_brut}' est introuvable dans le fichier HDF5.")
        
        dataset_source = f_source[chemin_dataset_brut]
        shape_source = dataset_source.shape
        print(f"Shape du signal brut original : {shape_source}") # Exemple: (2336800, 1700)
        
        # Structure attendue par le dataset : (Time x Channels)
        # On extrait la slice correspondant aux canaux choisis (incluant la borne fin)
        print("Extraction et copie des données en cours (cela peut prendre un moment)...")
        
        # Extraction de la plage [canal_debut : canal_fin + 1]
        donnees_filtrees = dataset_source[:, canal_debut:canal_fin + 1]
        
        # 3. Recréer l'arborescence des groupes et sauvegarder les données filtrées
        # Copie également des métadonnées temporelles associées si besoin
        f_dest.create_dataset(chemin_dataset_brut, data=donnees_filtrees, compression="gzip")
        
        # Optionnel : Copier aussi le vecteur de temps s'il existe pour garder le fichier valide
        chemin_temps = "/Acquisition/Raw[0]/RawDataTime"
        if chemin_temps in f_source:
            f_dest.create_dataset(chemin_temps, data=f_source[chemin_temps][:])
            
        print(f"Copie réussie ! Nouveau shape : {donnees_filtrees.shape}")
        print(f"Fichier sauvegardé sous : {chemin_h5_sortie}")

# --- CONFIGURATION DES CHEMINS ---
# Remplacez par vos chemins réels
chemin_du_txt = r"D:\Telechargement\DAS-dataset\data\walk\walking2_2023-04-17T122153+0100.txt"
chemin_du_h5 = r"D:\Telechargement\DAS-dataset\data\walk\walking2_2023-04-17T122153+0100.h5"
dossier_de_sortie = r"D:\Telechargement\DAS_dataset_active\data\walk"

# Lancement du script
copier_canaux_actifs(chemin_du_h5, chemin_du_txt, dossier_de_sortie)