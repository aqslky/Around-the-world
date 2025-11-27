# Around the World
This repository explores a graph-based shortest-path problem using world city data. The project builds a graph where each city connects to its 3 nearest neighbors and evaluates whether a traveler starting in London and always heading east can circumnavigate the globe and return within 80 days.

**Dataset**
- Source: Kaggle - World Cities / `worldcitiespop.csv` (see `data/raw/`).
- Processed CSVs are in `data/processed/` (e.g. `worldcities_processed.csv`, `worldcities_processed_major.csv`).

**Rules used to build the graph**
- Each city connects to its 3 geographically nearest neighbor cities.
- Base travel times: nearest=2h, 2nd=4h, 3rd=8h.
- Add +2h if moving to a different country.
- Add +2h if destination population > 200,000.

**Repository layout**
- `data/raw/` : original dataset files (not tracked in repo if `.gitignore` used).
- `data/processed/` : cleaned and preprocessed CSVs used by notebooks.
- `notebooks/` : exploratory and analysis notebooks (`01_data_preprocessing.ipynb`, `02_constructing_graph.ipynb`, `03_finding_shortest_path.ipynb`).
- `src/` : helper modules used by notebooks (e.g. `utils.py`).

**Dependencies**
- See `requirements.txt` for Python packages. Typical environment:

	- Python 3.8+ (recommended)
	- `pandas`, `networkx`, `geopy`, `numpy`, `matplotlib`, `jupyter`

**Quick setup (Linux / macOS)**
1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open notebooks:

```bash
jupyter lab notebooks
```

**How to run the analysis**
- Run the notebooks in order: `01_data_preprocessing.ipynb` → `02_constructing_graph.ipynb` → `03_finding_shortest_path.ipynb`.
- Notebooks use the `data/processed/` CSVs; re-run preprocessing if you change raw inputs.

**Notes:**
Please, refer to the markdown comments in the notebooks for the final logic of the project and the reasoning behind it