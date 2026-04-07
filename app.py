from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
	page_title="NASA Turbofan Offline RL",
	page_icon="🛠️",
	layout="wide",
)


@st.cache_data
def load_cmapps_file(file_path: Path) -> pd.DataFrame:
	"""Load a C-MAPSS text file (space-separated, no header)."""
	# C-MAPSS files have trailing spaces; sep='\s+' handles variable spacing.
	raw_df = pd.read_csv(file_path, sep=r"\s+", header=None)

	# Remove empty trailing columns if present.
	raw_df = raw_df.dropna(axis=1, how="all")

	if raw_df.shape[1] != 26:
		return raw_df

	columns = ["unit", "cycle"]
	columns += [f"op_setting_{i}" for i in range(1, 4)]
	columns += [f"sensor_{i}" for i in range(1, 22)]

	raw_df.columns = columns
	return raw_df


def main() -> None:
	st.title("NASA Turbofan - Base Streamlit")
	st.caption("Point de départ pour visualiser les données C-MAPSS et préparer le pipeline RL.")

	data_dir = Path("data") / "CMaps"

	st.sidebar.header("Configuration")
	dataset = st.sidebar.selectbox(
		"Jeu de données",
		["FD001", "FD002", "FD003", "FD004"],
		index=0,
	)
	split = st.sidebar.radio("Split", ["train", "test"], horizontal=True)
	selected_file = data_dir / f"{split}_{dataset}.txt"

	if not selected_file.exists():
		st.error(f"Fichier introuvable: {selected_file}")
		st.stop()

	df = load_cmapps_file(selected_file)

	left, right = st.columns([2, 1])
	with left:
		st.subheader("Aperçu des données")
		st.dataframe(df.head(20), use_container_width=True)

	with right:
		st.subheader("Infos rapides")
		st.metric("Lignes", f"{len(df):,}".replace(",", " "))
		st.metric("Colonnes", df.shape[1])
		if "unit" in df.columns:
			st.metric("Moteurs uniques", int(df["unit"].nunique()))

	st.subheader("Distribution des cycles par moteur")
	if {"unit", "cycle"}.issubset(df.columns):
		cycles = df.groupby("unit", as_index=False)["cycle"].max()
		st.bar_chart(cycles.set_index("unit")["cycle"])
	else:
		st.info("Colonnes attendues non détectées (format brut affiché).")

	st.divider()
	st.markdown("### Prochaines étapes")
	st.markdown("- Ajouter le calcul de RUL")
	st.markdown("- Préparer les features pour entraînement")
	st.markdown("- Intégrer une section d'entraînement/inférence RL")


if __name__ == "__main__":
	main()
