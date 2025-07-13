import fluoriclogppka
from fluoriclogppka.ml_part.services.molecule_3d_features_service import Molecule3DFeaturesService

def get_3d_features(smiles: str, target_value) -> dict:
    """Отримує 3D характеристики молекули"""
    try:
        service = Molecule3DFeaturesService(
            smiles=smiles,
            target_value=target_value,
            conformers_limit=None
        )
        
        return service.features_3d_dict
    except Exception as e:
        return None

def display_3d_features(features_dict: dict):
    """Відображає 3D характеристики молекули в зручному форматі"""
    if not features_dict:
        return
    
    # Групуємо характеристики за категоріями
    basic_features = {
        "Ідентифікатор": features_dict.get("identificator"),
        "Дипольний момент": features_dict.get("dipole_moment"),
        "Об'єм молекули": features_dict.get("mol_volume"),
        "Молекулярна вага": features_dict.get("mol_weight"),
        "SASA": features_dict.get("sasa"),
        "TPSA+F": features_dict.get("tpsa+f")
    }
    
    angle_features = {
        "Кут X1X2R2": features_dict.get("angle_X1X2R2"),
        "Кут X2X1R1": features_dict.get("angle_X2X1R1"),
        "Кут R2X2R1": features_dict.get("angle_R2X2R1"),
        "Кут R1X1R2": features_dict.get("angle_R1X1R2"),
        "Двогранний кут": features_dict.get("dihedral_angle")
    }
    
    distance_features = {
        "F до FG": features_dict.get("f_to_fg"),
        "F свобода": features_dict.get("f_freedom"),
        "Відстань між атомами в циклі": features_dict.get("distance_between_atoms_in_cycle_and_f_group"),
        "Відстань між центрами F-груп": features_dict.get("distance_between_atoms_in_f_group_centers"),
        "Cis/Trans": features_dict.get("cis/trans")
    }

    return basic_features, angle_features, distance_features
    

if __name__ == "__main__":

    SMILES = "F[C@H]1C[C@H](F)CN(C1)C(=O)C1=CC=CC=C1"

    inference = fluoriclogppka.Inference(SMILES=SMILES,
                                         target_value=fluoriclogppka.Target.logP)
    
    print(inference.predict())

    features_3d = get_3d_features(SMILES, fluoriclogppka.Target.logP)
    print(display_3d_features(features_3d))
