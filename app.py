import streamlit as st
import fluoriclogppka
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import io
import base64
import tempfile
import os
import pandas as pd
from typing import Optional
from streamlit_ketcher import st_ketcher

# Налаштування сторінки
st.set_page_config(
    page_title="Fluoriclogppka Prediction",
    page_icon="🧪",
    layout="wide"
)

def sdf_to_smiles(sdf_content: str) -> Optional[str]:
    """Конвертує SDF файл в SMILES"""
    try:
        mol_supplier = Chem.SDMolSupplier()
        mol_supplier.SetData(sdf_content)
        
        for mol in mol_supplier:
            if mol is not None:
                return Chem.MolToSmiles(mol)
        return None
    except Exception as e:
        st.error(f"Помилка при обробці SDF файлу: {str(e)}")
        return None

def draw_molecule(smiles: str) -> str:
    """Малює молекулу та повертає HTML"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "<p>Не вдалося намалювати молекулу</p>"
        
        # Створюємо зображення молекули
        img = Draw.MolToImage(mol, size=(400, 400))
        
        # Конвертуємо в base64 для відображення в HTML
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f'<img src="data:image/png;base64,{img_str}" style="max-width: 100%; height: auto;">'
    except Exception as e:
        return f"<p>Помилка при малюванні молекули: {str(e)}</p>"

def get_2d_features(smiles: str) -> dict:
    """Отримує 2D характеристики молекули"""
    try:
        from fluoriclogppka.ml_part.services.molecule_2d_features_service import Molecule2DFeaturesService
        
        service = Molecule2DFeaturesService(
            SMILES=smiles
        )
        
        return service.features_2d_dict
    except Exception as e:
        st.error(f"Помилка при розрахунку 2D характеристик: {str(e)}")
        return None

def get_3d_features(smiles: str, target_value) -> dict:
    """Отримує 3D характеристики молекули"""
    try:
        from fluoriclogppka.ml_part.services.molecule_3d_features_service import Molecule3DFeaturesService
        
        service = Molecule3DFeaturesService(
            smiles=smiles,
            target_value=target_value,
            conformers_limit=None
        )
        
        return service.features_3d_dict
    except Exception as e:
        st.error(f"Помилка при розрахунку 3D характеристик: {str(e)}")
        return None

def display_2d_features(features_dict: dict):
    """Відображає 2D характеристики молекули в одному горизонтальному рядку"""
    if not features_dict:
        return

    st.subheader("🧬 2D Характеристики молекули")
    st.markdown("**Основні характеристики:**")

    # Створюємо колонки — по одній на кожну ознаку
    cols = st.columns(len(features_dict))

    for col, (name, value) in zip(cols, features_dict.items()):
        if value is not None:
            with col:
                # Виводимо назву ознаки й значення в одному стилі
                if isinstance(value, (int, float)):
                    st.metric(name, f"{str(value)}" if isinstance(value, float) else str(value))
                else:
                    st.metric(name, str(value))
    
    with st.expander("Детальна таблиця всіх характеристик"):
        df_data = []
        for key, value in features_dict.items():
            if value is not None:
                df_data.append({
                    "Характеристика": key,
                    "Значення": value if not isinstance(value, float) else f"{str(value)}",
                })
        
        if df_data:
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Немає доступних характеристик для відображення")

def display_3d_features(features_dict: dict):
    """Відображає 3D характеристики молекули в зручному форматі"""
    if not features_dict:
        return
    
    st.subheader("🧬 3D Характеристики молекули")
    
    # Створюємо кілька колонок для кращого відображення
    col1, col2, col3 = st.columns(3)
    
    # Групуємо характеристики за категоріями
    basic_features = {
        # "Ідентифікатор": features_dict.get("identificator"),
        # "Дипольний момент": features_dict.get("dipole_moment"),
        "Об'єм молекули": features_dict.get("mol_volume"),
        "Молекулярна вага": features_dict.get("mol_weight"),
        "SASA": features_dict.get("sasa"),
        # "TPSA+F": features_dict.get("tpsa+f")
    }
    
    angle_features = {
        "Кут X1X2R2": features_dict.get("angle_X1X2R2"),
        "Кут X2X1R1": features_dict.get("angle_X2X1R1"),
        # "Кут R2X2R1": features_dict.get("angle_R2X2R1"),
        # "Кут R1X1R2": features_dict.get("angle_R1X1R2"),
        "Двогранний кут": features_dict.get("dihedral_angle")
    }
    
    distance_features = {
        "F до FG": features_dict.get("f_to_fg"),
        # "F свобода": features_dict.get("f_freedom"),
        "Відстань між атомами в циклі": features_dict.get("distance_between_atoms_in_cycle_and_f_group"),
        "Відстань між центрами F-груп": features_dict.get("distance_between_atoms_in_f_group_centers"),
        # "Cis/Trans": features_dict.get("cis/trans")
    }
    
    with col1:
        st.markdown("**Основні характеристики:**")
        for name, value in basic_features.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    st.metric(name, f"{value:.4f}" if isinstance(value, float) else str(value))
                else:
                    st.metric(name, str(value))
    
    with col2:
        st.markdown("**Кути:**")
        for name, value in angle_features.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    st.metric(name, f"{value:.4f}°" if isinstance(value, float) else f"{value}°")
                else:
                    st.metric(name, str(value))
    
    with col3:
        st.markdown("**Відстані та конформація:**")
        for name, value in distance_features.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    st.metric(name, f"{value:.4f}" if isinstance(value, float) else str(value))
                else:
                    st.metric(name, str(value))
    
    # Додаємо детальну таблицю з усіма характеристиками
    with st.expander("Детальна таблиця всіх характеристик"):
        df_data = []
        for key, value in features_dict.items():
            if value is not None:
                df_data.append({
                    "Характеристика": key,
                    "Значення": value if not isinstance(value, float) else f"{value:.6f}",
                })
        
        if df_data:
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Немає доступних характеристик для відображення")

def validate_smiles(smiles: str) -> bool:
    """Перевіряє валідність SMILES рядка"""
    if not smiles or not isinstance(smiles, str):
        return False
    
    # Видаляємо пробіли з початку і кінця
    smiles = smiles.strip()
    
    # Перевіряємо, чи не порожній рядок
    if not smiles:
        return False
    
    try:
        # Спробуємо створити молекулу з SMILES
        mol = Chem.MolFromSmiles(smiles)
        
        # Перевіряємо, чи молекула створена успішно
        if mol is None:
            return False
        
        # Додаткові перевірки
        # Перевіряємо, чи є атоми в молекулі
        if mol.GetNumAtoms() == 0:
            return False
        
        # Перевіряємо, чи можна створити канонічний SMILES
        canonical_smiles = Chem.MolToSmiles(mol)
        if not canonical_smiles:
            return False
        
        # Спробуємо створити молекулу з канонічного SMILES (додаткова перевірка)
        test_mol = Chem.MolFromSmiles(canonical_smiles)
        if test_mol is None:
            return False
        
        return True
        
    except Exception as e:
        # Якщо виникла будь-яка помилка, SMILES невалідний
        return False

# Основний інтерфейс
st.title("🧪 Fluoriclogppka Prediction")
st.markdown("Додаток для передбачення властивостей молекул за допомогою fluoriclogppka")

# Радіобаттони для вибору способу введення молекули
input_method_choice = st.radio(
    "Оберіть спосіб введення молекули:",
    ("📝 Введення SMILES", "📁 Завантаження SDF", "🎨 Редактор молекул"),
    index=0,
    horizontal=True
)

# Ініціалізуємо змінні
final_smiles = None
input_method = None
smiles_input = None
uploaded_file = None
editor_smiles = None

# Показуємо відповідний інтерфейс залежно від вибору
if input_method_choice == "📝 Введення SMILES":
    st.header("📝 Введення SMILES")
    smiles_input = st.text_area(
        "Введіть SMILES рядок:",
        value="FC1(F)CC(C(O)=O)C1",
        height=70,
        help="Введіть SMILES представлення молекули"
    )
    
    if smiles_input and validate_smiles(smiles_input):
        final_smiles = smiles_input
        input_method = "Введення SMILES"

elif input_method_choice == "📁 Завантаження SDF":
    st.header("📁 Завантаження SDF")
    uploaded_file = st.file_uploader(
        "Завантажте SDF файл:",
        type=['sdf'],
        help="Завантажте файл з молекулою в форматі SDF"
    )
    
    if uploaded_file is not None:
        try:
            sdf_content = uploaded_file.read().decode('utf-8')
            sdf_smiles = sdf_to_smiles(sdf_content)
            if sdf_smiles:
                final_smiles = sdf_smiles
                input_method = "SDF файл"
        except Exception as e:
            st.error(f"Помилка при читанні SDF файлу: {str(e)}")

elif input_method_choice == "🎨 Редактор молекул":
    st.header("🎨 Редактор молекул")
    
    # Використовуємо streamlit-ketcher для малювання молекул
    molecule = st_ketcher(
        height=400,
        key="molecule_editor"
    )
    
    # Обробляємо результат з редактора
    if molecule:
        try:
            # Різні варіанти структури даних, які може повернути streamlit-ketcher
            editor_smiles = None
            
            if hasattr(molecule, 'smiles') and molecule.smiles:
                editor_smiles = molecule.smiles
            elif isinstance(molecule, str):
                # Якщо повертається рядок, перевіряємо чи це SMILES
                if validate_smiles(molecule):
                    editor_smiles = molecule
            elif isinstance(molecule, dict):
                # Якщо повертається словник, шукаємо SMILES
                if 'smiles' in molecule:
                    editor_smiles = molecule['smiles']
                elif 'molfile' in molecule:
                    # Конвертуємо molfile в SMILES
                    try:
                        mol = Chem.MolFromMolBlock(molecule['molfile'])
                        if mol:
                            editor_smiles = Chem.MolToSmiles(mol)
                    except:
                        pass
            
            if editor_smiles and validate_smiles(editor_smiles):
                final_smiles = editor_smiles
                input_method = "Редактор молекул"
                st.success(f"Отримано SMILES з редактора: {editor_smiles}")
            else:
                st.info("Намалюйте молекулу в редакторі вище")
                
        except Exception as e:
            st.error(f"Помилка при обробці молекули з редактора: {str(e)}")
    else:
        st.info("Намалюйте молекулу в редакторі вище")

# Відображення поточної молекули
if final_smiles:
    # st.success(f"Використовується молекула з: {input_method}")
    # st.code(f"SMILES: {final_smiles}")
    
    # Малюємо молекулу
    mol_html = draw_molecule(final_smiles)
    st.markdown(mol_html, unsafe_allow_html=True)

# Параметри для inference
st.header("⚙️ Параметри передбачення")

col_params1, col_params2 = st.columns(2)

with col_params1:
    # Target value
    target_options = {
        "pKa": "pKa",
        "logP": "logP"  # Припускаємо, що є інші опції
    }
    target_value = st.selectbox(
        "Цільова властивість:",
        options=list(target_options.keys()),
        help="Оберіть властивість для передбачення"
    )
    
    # Model type
    model_options = {
        "GNN": "gnn",
        "H2O": "h2o"
    }
    model_type = st.selectbox(
        "Тип моделі:",
        options=list(model_options.keys()),
        help="Оберіть тип моделі для передбачення"
    )

# Кнопка для запуску передбачення
if st.button("🚀 Запустити передбачення", type="primary"):
    if final_smiles:
        try:
            with st.spinner("Виконується передбачення..."):
                # Створюємо об'єкт inference
                inference_params = {
                    "SMILES": final_smiles,
                    "target_value": getattr(fluoriclogppka.Target, target_value),
                    "model_type": getattr(fluoriclogppka.ModelType, model_options[model_type])
                }
                
                inference = fluoriclogppka.Inference(**inference_params)
                
                # Виконуємо передбачення
                result = inference.predict()
                
                # Відображаємо результат
                st.success("Передбачення успішно виконано!")
                st.subheader("📊 Результат передбачення:")
                
                if isinstance(result, dict):
                    for key, value in result.items():
                        st.metric(key, value)
                else:
                    print(result)
                    s = f"<p style='font-size:30px;'>{target_value}: {result}</p>"
                    st.markdown(s, unsafe_allow_html=True) 

                    features_2d = get_2d_features(final_smiles)

                    display_2d_features(features_2d)

                    features_3d = get_3d_features(final_smiles, getattr(fluoriclogppka.Target, target_value))

                    display_3d_features(features_3d)
                
                # Зберігаємо результат в session state
                st.session_state['last_prediction'] = {
                    'smiles': final_smiles,
                    'result': result,
                    'parameters': {**inference_params, **features_2d, **features_3d}
                }
                
        except Exception as e:
            st.error(f"Помилка при виконанні передбачення: {str(e)}")
    else:
        st.error("Будь ласка, введіть валідну молекулу")

# Розділ з історією передбачень
if 'last_prediction' in st.session_state:
    st.header("📈 Остання передбачення")
    with st.expander("Показати деталі"):
        st.json(st.session_state['last_prediction'])
    
    # Показуємо 3D характеристики, якщо вони є
    if '3d_features' in st.session_state['last_prediction']:
        display_3d_features(st.session_state['last_prediction']['3d_features'])
