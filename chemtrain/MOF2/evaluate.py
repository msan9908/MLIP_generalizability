import os
from evaluate_models import evaluate_model
from evaluate_models import evaluate_cluster_or_maxsep_model
import sys
if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
model_paths = {
    "Mace": {
        "MP4 SL noC": "output/Qeq_MOF_MP4_SL_noc_MACE_2025_7_7_656a21cd-b360-4d1f-ada6-d9b2f26ae782",
        "MP4 maxsep noC": "output/Qeq_Qeq_MOF_MACE_maxsep_noC_MACE_2025_7_2_f8ad5055-7f7f-42f8-9596-988181f7d0f5",
        "MP4 cluster noC": "output/Qeq_Qeq_MOF_MACE_cluster_noC_MACE_2025_7_14_80253fcb-2bf7-4b94-843f-6bb33e5fd397",
        "SL noC": "output/Qeq_Qeq_MOF_MACE_SL_noC_MACE_2025_6_17_853f6e72-938e-4391-b1d2-b7336e0fab04",
        "maxsep noC": "output/Qeq_Qeq_MOF_MACE_maxsep_noC_MACE_2025_6_20_753ddec5-4fb3-4b77-b097-fa96853e020a",
        "cluster noC": "output/Qeq_Qeq_MOF_MACE_cluster_noC_MACE_2025_6_20_3fe4a5fd-3d7a-4ada-8f58-965818352f37",
        "SL C": "output/Qeq_Qeq_MOF_MACE_SL_C_MACE_2025_6_23_782ce1e3-60e9-4a56-8892-4d5e75cf1158",
        "maxsep C": "output/Qeq_Qeq_MOF_MACE_maxsep_C_MACE_2025_6_23_cc1b2b23-ff7c-4822-991c-39c29b52fff8",
        "cluster C": "output/Qeq_Qeq_MOF_MACE_cluster_C_MACE_2025_6_16_e14dafa0-c164-4608-94ae-235342249d95",
        "MP4 rand noC": "output/Qeq_MACE_MP4_rand_noC_MACE_2025_9_23_9ec48525-92ef-4075-bef4-a93b3918e4d5",
        "rand C": "output/Qeq_MACE_rand_C_MACE_2025_9_24_24411eae-57a2-4da7-9f0c-25af550d375c",
        "rand noC": "Qeq_MACE_rand_noC_MACE_2025_9_24_f05e1e08-dbbb-41fa-83e5-9f106061a12f",
        "rand EFA": "Qeq_MACE_EFA_rand_MACE_EFA_2025_9_25_fba1260b-cfb9-4226-8816-3891aa2e800c",
        "SL EFA": "output/Qeq_Qeq_MOF_painn_SL_MACE_EFA_2025_7_4_ba79757d-d10f-4570-80ec-1d410736840f",
        "maxsep EFA": "output/Qeq_Qeq_MOF_MACE_maxsep_noC_MACE_EFA_2025_7_4_73cb1147-7f94-4369-ad9d-f7b97b8a6f91",
        "cluster EFA": "output/Qeq_MACE_clus_noC_MACE_EFA_2025_7_12_cfeaab58-9219-462c-b395-7ee20be0d839",
    },

    "Allegro": {

        "SL noC": "output/Qeq_Allegro_MOF_SL_AllegroQeq_2025_6_18_44c6c0ed-0969-42c4-8006-9209f0bfecef",
        "maxsep noC": "output/Qeq_MAXSEP_Allegro_AllegroQeq_2025_6_18_d96e08b5-7d26-4e54-9fab-b4fd34ff4daa",
        "cluster noC": "output/Qeq_clus_Allegro_AllegroQeq_2025_6_18_9a7f02de-8314-4d42-b0a9-5714e449ac3c",
        "SL C": "output/Qeq_Allegro_MOF_SL_AllegroQeq_2025_6_18_44c6c0ed-0969-42c4-8006-9209f0bfecef",
        "maxsep C": "output/Qeq_Qeq_Allegro_C_maxsep_AllegroQeq_2025_7_10_57b123df-6807-474c-8b4f-e79d1fe52c70",
        "cluster C": "output/Qeq_Qeq_Allegro_clus_C_AllegroQeq_2025_7_10_e9d6edec-ad27-4967-b0f1-b2d2b248356f",

        "EFA SL noC" : 'output/Qeq_Qeq_MOF_painn_SL_EFA_2025_8_8_94027e0c-e603-48da-8a0c-77dc6bda3681',
        "EFA maxsep noC" : 'output/Qeq_EFA_MOF_maxsep_EFA_2025_8_8_abb12a90-7808-42d2-84c6-a9193e5eb745',
        "EFA cluster noC" : 'output/Qeq_EFA_MOF_clus_EFA_2025_8_8_ec8e7240-468f-4d1f-8623-4de6be64460d',
        "rand noC": "output/Qeq_allegro_noc_rand_AllegroQeq_2025_9_25_a295a96f-57c1-4655-9d2e-d4d5bbd66eea",
        "rand C": "output/Qeq_allegro_C_rand_AllegroQeq_2025_9_26_acb9cc4f-3364-4f1e-b24a-927e25f53a47",
        "rand EFA noC": "output/Qeq_allegro_EFA_rand_EFA_2025_9_25_d7717068-33e1-469c-82e9-883ee9660b37",

    },
        "Dimnet": {

        "SL noC": "../MOF/output/Qeq_Dimnet_SL_DimeNetPP_2025_7_31_c3b42e11-8e02-4a55-b696-af6903391c73",
        "maxsep noC": "../MOF/output/Qeq_maxsep_dimnet_noc_DimeNetPP_2025_7_30_12c24c8b-911b-4938-8292-1de9d78f7b94",
        "cluster noC": "../MOF/output/Qeq_clus_dimnet_noc_DimeNetPP_2025_7_30_f74fc495-dfca-45cd-8a33-526c59041f17",
        "rand noC": "output/Qeq_dimnet_rand_DimeNetPP_2025_9_26_58768626-41ab-4f4d-bf67-45b04c7771ec",

   },
}




dataset_path = "./"
save_root = "results/"

for model_family, family_models in model_paths.items():
    for model_variant, model_dir in family_models.items():
        save_path = os.path.join(save_root, model_family, model_variant.replace(" ", "_"))
        os.makedirs(save_path, exist_ok=True)

        print(f"🔍 Evaluating [{model_family}] - [{model_variant}]")

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



