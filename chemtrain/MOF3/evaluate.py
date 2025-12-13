import os
from evaluate_models import evaluate_model
from evaluate_models import evaluate_cluster_or_maxsep_model
import sys
if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


model_paths = {
    "MACE": {
        "cluster C": "output/Qeq_Qeq_MOF_MACE_cluster_C_MACE_2025_9_3_be7985e1-9f74-4b14-8bd2-5bf4e21d8794",
        "maxsep C": "output/Qeq_Qeq_MOF_MACE_maxsep_C_MACE_2025_9_4_59aaebeb-f24f-4cb3-9c5a-ed7a506ac3b2",
        "maxsep noC": "output/Qeq_Qeq_MOF_MACE_maxsep_noC_MACE_2025_9_4_6b0e4e81-6cb3-4fd5-abc9-88dc55584f2e",
        "cluster noC": "output/Qeq_Qeq_MOF_MACE_cluster_noC_MACE_2025_9_5_ab574577-3346-4288-970a-00975c2a3334",
        "cluster EFA": "output/Qeq_MACE_clus_noC_MACE_EFA_2025_9_8_de5a3a78-171d-4000-8b71-bba83db82a98",
        "maxsep EFA": "output/Qeq_Qeq_MOF_MACE_maxsep_noC_MACE_EFA_2025_9_8_bb98c826-1ae1-4967-89b9-0f5055baf040",
        "MP4 maxsep noC": "output/Qeq_MACE_MP4_maxsep_noC_MACE_2025_9_10_0af05088-f8ea-4691-b020-20ef523c3e81",
        "MP4 cluster noC": "output/Qeq_Qeq_MOF_MACE_cluster_noC_MACE_2025_9_11_40156521-6f37-4b68-b92f-9dffe5446ac7",
        "MP4 SL noC": "output/Qeq_MOF_MP4_SL_noc_MACE_2025_9_12_f861de63-3a35-426e-92ea-c630c867cbc8",
        "SL C": "output/Qeq_MACE_SL_C_MACE_2025_9_13_07fd2c3f-6d01-448d-a7be-3b6699a5c588",
        "SL noC": "output/Qeq_Qeq_MOF_MACE_SL_noC_MACE_2025_9_13_887b295c-9e21-4fb5-86e2-4d6652509bf5",
        "SL EFA": "output/Qeq_Qeq_MOF_painn_SL_MACE_EFA_2025_9_22_72f425b6-c9e3-4539-800a-4d85dcae7bfd",
        "MP4 rand noC": "output/Qeq_MACE_MP4_rand_noC_MACE_2025_9_17_8bfa0f77-954c-424f-8c18-1c62a9d55e7b",
        "rand C": "output/Qeq_MACE_rand_C_MACE_2025_9_18_cc990a0e-f9f3-4548-88a3-8914824ac749",
        "rand noC": "output/Qeq_MACE_rand_noC_MACE_2025_9_18_17ff5ad9-927b-4814-8b78-23d2ff874f5d",
        "rand EFA": "output/Qeq_MACE_EFA_rand_MACE_EFA_2025_9_21_41d26847-5ddd-47c3-91f9-cc4df90a0c55",
    },

    "Allegro": {
        "cluster noC": "output/Qeq_clus_Allegro_AllegroQeq_2025_9_6_57682c88-9869-4bb5-be1f-3ae729bb4448",
        "maxsep noC": "output/Qeq_MAXSEP_Allegro_AllegroQeq_2025_9_7_e39a5141-62cf-4ed1-9689-f2a035d26faf",
        "SL noC": "output/Qeq_Allegro_MOF_SL_AllegroQeq_2025_9_14_0ea923c9-8b99-49d9-afe2-0ab6474e8a0d",
        "SL C": "output/Qeq_Qeq_MOF_AllegroQeq_2025_9_14_89e750f2-7fc7-40e6-94ac-6b6cf85592c2",
        "rand noC": "output/Qeq_allegro_noc_rand_AllegroQeq_2025_9_19_ab365625-0fec-482b-b11f-a6e8f0e8a778",
        "rand C": "output/Qeq_allegro_C_rand_AllegroQeq_2025_9_20_5c022562-2dd2-4e33-9213-93355c420134",
        "EFA SL": "output/Qeq_Qeq_MOF_painn_SL_EFA_2025_9_13_d9447478-b6db-4e4f-999d-2b9c5ccc064a",
        "maxsep EFA noC": "output/Qeq_EFA_MOF_maxsep_EFA_2025_9_21_826ea7c2-fea4-49f0-a9e7-84566b8c8242",
        "cluster EFA noC": "output/Qeq_EFA_MOF_clus_EFA_2025_9_5_0a6379b5-a57c-4498-8a5d-428e59e2d0d0",
        "rand EFA noC": "output/Qeq_allegro_EFA_rand_EFA_2025_9_19_bd2c045e-fea5-43bd-ada0-025da8e476bc",
        "maxsep C": "output/Qeq_Qeq_Allegro_C_maxsep_AllegroQeq_2025_9_7_5b169e3b-58d7-4bc8-8a14-a44606f4212a",
        "cluster C": "output/Qeq_Qeq_Allegro_clus_C_AllegroQeq_2025_9_7_584686a9-cb9f-4ece-8b84-843cf02d48e5",
    },
    "Dimnet": {
        "SL C": "output/Qeq_SL_dimnet_DimeNetPP_2025_9_14_2480a0fa-7a05-4f02-b81c-90acb4c04a8e",
        "cluster C": "output/Qeq_clus_dimnet_DimeNetPP_2025_9_15_eaaeaa7e-eb8b-4fab-aa3c-4e1339ce90e4",
        "maxsep C": "output/Qeq_maxsep_dimnet_DimeNetPP_2025_9_16_be26b639-599c-4c81-b9cb-35bbf0040266",
        "rand C": "output/Qeq_dimnet_rand_DimeNetPP_2025_9_20_42ba420d-63b0-4431-b3df-70b2f724c163",
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
                    use_coulomb_nbrs=use_coulomb_nbrs)
            else:
                print(f"⚠️ Skipping model {model_variant} (not SL / cluster / maxsep)")
        except Exception as e:
            print(f"❌ Failed on {model_family} - {model_variant}: {e}")



