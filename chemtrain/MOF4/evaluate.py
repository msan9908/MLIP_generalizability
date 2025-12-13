import os
from evaluate_models import evaluate_model
from evaluate_models import evaluate_cluster_or_maxsep_model
import sys
if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


model_paths = {
    "MACE": {
        "cluster C": "Qeq_Qeq_MOF_MACE_cluster_C_MACE_2025_9_4_772bb7ba-d96a-4fc6-b0a0-e9f8237305ae",
        "maxsep C": "Qeq_Qeq_MOF_MACE_maxsep_C_MACE_2025_9_5_d097648a-81da-4764-b5f3-dec5d4f3f871",
        "maxsep noC": "Qeq_Qeq_MOF_MACE_maxsep_noC_MACE_2025_9_5_a1e8ab88-f46e-4da5-ae78-ed93b804e6f9",
        "cluster noC": "Qeq_Qeq_MOF_MACE_cluster_noC_MACE_2025_9_6_4fcdd500-1f31-4784-8918-777a65eea559",
        "cluster EFA": "Qeq_MACE_clus_noC_MACE_EFA_2025_9_8_137a3bf1-8bd8-440d-8835-4025d698bffc",
        "maxsep EFA": "Qeq_Qeq_MOF_MACE_maxsep_noC_MACE_EFA_2025_9_9_1e9de679-f2fa-4632-bfdd-2c08af682a95",
        "MP4 maxsep noC": "Qeq_MACE_MP4_maxsep_noC_MACE_2025_9_10_fc13b8f8-dec2-48bc-90bc-37b773a29f5a",
        "MP4 cluster noC": "Qeq_Qeq_MOF_MACE_cluster_noC_MACE_2025_9_11_2633feb5-4ca6-4e65-bea6-8d1939b73153",
        "MP4 SL noC": "Qeq_MOF_MP4_SL_noc_MACE_2025_9_12_41264b77-e480-4e56-9cfb-c5318420690d",
        "SL C": "Qeq_MACE_SL_C_MACE_2025_9_12_c2aa8f68-e3c4-4800-bc56-41105444463d",
        "SL noC": "Qeq_Qeq_MOF_MACE_SL_noC_MACE_2025_9_13_3b77e0cc-4980-4e8b-82bf-65840bf460a8",
        "SL EFA": "Qeq_Qeq_MOF_painn_SL_MACE_EFA_2025_9_22_f50c90b5-0c6d-4be8-8107-4b009dfeb782",
        "MP4 rand noC": "Qeq_MACE_MP4_rand_noC_MACE_2025_9_17_ed94a7ef-98fb-4c2e-a4ea-c8df7197bf0e",
        "rand C": "Qeq_MACE_rand_C_MACE_2025_9_18_20f9a8b9-da96-472d-8f1e-ca2acfc85b09",
        "rand noC": "Qeq_MACE_rand_noC_MACE_2025_9_18_33bbf62c-5ba9-446e-99fd-be205c75d825",
        "rand EFA": "Qeq_MACE_EFA_rand_MACE_EFA_2025_9_21_f8c15a47-4f5b-4ec9-beb7-1c6abd4ccb28",
    },

    "Allegro": {
        "cluster noC": "Qeq_clus_Allegro_AllegroQeq_2025_9_7_98f2438a-4d11-40a8-8115-8a4a69a5f2d8",
        "maxsep noC": "Qeq_MAXSEP_Allegro_AllegroQeq_2025_9_7_562cff9a-09d1-4d5a-acc4-7515309bb51d",
        "maxsep C": "Qeq_Qeq_Allegro_C_maxsep_AllegroQeq_2025_9_8_f8ca5f4c-ccec-4175-80ef-81c862d8f0c9",
        "cluster C": "Qeq_Qeq_Allegro_clus_C_AllegroQeq_2025_9_8_a9a9e2d9-67f7-47a4-b982-f72d5e5fc41d",
        "SL EFA": "Qeq_Qeq_MOF_painn_SL_EFA_2025_9_13_f40c813a-7e9e-4d49-bc56-b27b7342582c",
        "SL noC": "Qeq_Allegro_MOF_SL_AllegroQeq_2025_9_14_1167c03a-00e4-410e-9054-a82b3ad05d05",
        "rand noC": "Qeq_allegro_noc_rand_AllegroQeq_2025_9_19_27995797-5a7d-491a-bed8-19e3cb8e2d08",
        "rand C": "Qeq_allegro_C_rand_AllegroQeq_2025_9_20_dcd914cf-f2ab-40ae-a472-a766978f4319",
        "SL C": "Qeq_Qeq_MOF_AllegroQeq_2025_9_14_59a21f4b-5583-4326-90ff-e3b2b6a37560",
        "maxsep EFA": "Qeq_EFA_MOF_maxsep_EFA_2025_9_21_58e1dcc8-ed7c-4bd0-b925-937a35e744cd",
        "cluster EFA": "Qeq_EFA_MOF_clus_EFA_2025_9_6_6116e90f-f6be-4745-9da4-123797640fe4",
        "rand EFA": "Qeq_allegro_EFA_rand_EFA_2025_9_19_9528656e-f8a1-4d4f-9123-618c967950c3",
    },

    "Dimnet": {
        "SL noC": "Qeq_SL_dimnet_DimeNetPP_2025_9_14_aa2d5263-e175-4f65-a02e-0ab47ab8591b",
        "cluster noC": "Qeq_clus_dimnet_DimeNetPP_2025_9_15_5ac0b5c7-37a2-4d3b-81f2-e25c214b5cba",
        "maxsep noC": "Qeq_maxsep_dimnet_DimeNetPP_2025_9_16_8ab5de67-8219-487d-b366-f5ff0b295a47",
        "rand noC": "Qeq_dimnet_rand_DimeNetPP_2025_9_20_6d3f9a52-cec6-4262-bad0-1d45db1ab7a0",
    },
}
    



dataset_path = "./"
save_root = "results/"

for model_family, family_models in model_paths.items():
    for model_variant, model_dir in family_models.items():
        save_path = os.path.join(save_root, model_family, model_variant.replace(" ", "_"))
        os.makedirs(save_path, exist_ok=True)

        print(f"🔍 Evaluating [{model_family}] - [{model_variant}]")
        model_dir = os.path.join('output', model_dir)

        try:
            use_coulomb_nbrs = " C" in model_variant # true for "SL C", "maxsep C", etc.

            if "SL" in model_variant:
                evaluate_model(
                    model_dir=model_dir,
                    save_path=save_path,
                    dataset_path=dataset_path,
                    use_coulomb_nbrs=use_coulomb_nbrs
                )
            elif "cluster" in model_variant.lower():
                dataset_file = os.path.join(dataset_path, "cluster_mof_dataset.npz")
                evaluate_cluster_or_maxsep_model(
                    model_dir=model_dir,
                    save_path=save_path,
                    dataset_path=dataset_file,
                    use_coulomb_nbrs=use_coulomb_nbrs
                )
            elif "maxsep" in model_variant.lower():
                dataset_file = os.path.join(dataset_path, "maxsep_mof_dataset.npz")
                evaluate_cluster_or_maxsep_model(
                    model_dir=model_dir,
                    save_path=save_path,
                    dataset_path=dataset_file,
                    use_coulomb_nbrs=use_coulomb_nbrs
                )
            elif "rand" in model_variant.lower():
                dataset_file = os.path.join(dataset_path, "rand_mof_dataset.npz")
                evaluate_cluster_or_maxsep_model(
                    model_dir=model_dir,
                    save_path=save_path,
                    dataset_path=dataset_file,
                    use_coulomb_nbrs=use_coulomb_nbrs
                )
            else:
                print(f"⚠️ Skipping model {model_variant} (not SL / cluster / maxsep)")
        except Exception as e:
            print(f"❌ Failed on {model_family} - {model_variant}: {e}")



