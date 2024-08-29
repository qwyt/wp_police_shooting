import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import random
import warnings
import utils.helper as helper

INCLUDE_CORR_DEMO_VARS = [
    'Persons under 5 years, percent',
    'Persons under 18 years, percent',
    'Persons 65 years and over, percent',
    'Female persons, percent',
    # Uncomment to include these variables into demographic analysis
    # 'White alone, percent',
    # 'Black or African American alone, percent',
    # 'American Indian and Alaska Native alone, percent',
    # 'Asian alone, percent',
    # 'Two or More Races, percent',
    # 'Hispanic or Latino, percent',
    # 'White alone, not Hispanic or Latino, percent',
    'Living in same house 1 year & over, percent',
    'Foreign born persons, percent',
    'Language other than English spoken at home, pct age 5+',
    'High school graduate or higher, percent of persons age 25+',
    "Bachelor's degree or higher, percent of persons age 25+",
    'Mean travel time to work (minutes), workers age 16+',
    'Homeownership rate',
    'Housing units in multi-unit structures, percent',
    'Median value of owner-occupied housing units',
    'Persons per household',
    'Per capita money income in past 12 months (2013 dollars)',
    'Median household income',
    'Persons below poverty level, percent',
    'Population per square mile, 2010']

DEMO_KEYS = {
    "All": "PST045214",
    "Black": "RHI225214",
    "White": "RHI125214",
    "Asian": "RHI425214",
    "Hispanic": "RHI725214",
    "Native American": "RHI325214",
}


def load_and_process_shootings_df():
    df_main = pd.read_csv("datasets/data-police-shootings/v2/fatal-police-shootings-data.csv")

    # Add age brackets
    df_main["age_bracket"] = pd.cut(df_main["age"], [0, 17, 24, 34, 49, 64, 999], labels=[
        "17 or younger",
        "18-24",
        "25-34",
        "35-49",
        "50-64",
        "65 or older"
    ])

    age_bracket_2_labels = [
        "25 or younger",
        "25-35",
        "35-45",
        "45+",
    ]
    df_main["age_bracket_short"] = pd.cut(df_main["age"], [0, 25, 35, 45, 9999], labels=age_bracket_2_labels)


    # Rename races
    race_map = {"W": "White", "B": "Black", "A": "Asian", "N": "Native American", "H": "Hispanic", "O": "Other",
                "B;H": "Other"}

    df_main["race"] = df_main["race"].map(race_map)

    # Load state and country level demographics data
    df_facts = pd.read_csv("datasets/county_facts/county_facts.csv")

    # Append full country facts
    _df_facts_columns = pd.read_csv("datasets/county_facts/county_facts_dictionary.csv")
    df_facts_columns = _df_facts_columns.set_index("column_name")

    def get_facts_df_readable(df):
        """
        Map each index column
        """
        return df.rename(columns=df_facts_columns.to_dict()["description"])

    df_main["county"] = df_main["county"].astype("string")

    df_main["stateCounty"] = df_main["state"].map(lambda s: f"{s}, ") + df_main["county"]
    df_main["stateCounty"] = df_main["stateCounty"].astype("string")

    df_main["stateCounty"] = df_main["stateCounty"].str.strip().str.lower()

    df_facts_only_count = df_facts[df_facts["area_name"].str.contains("County")]
    df_facts["stateCounty"] = df_facts["state_abbreviation"].map(lambda s: f"{s}, ") + df_facts["area_name"]. \
        map(lambda s: s.replace("County", "").replace("Parish", "").replace("Municipality", ""))
    df_facts["stateCounty"] = df_facts["stateCounty"].astype("string")
    df_facts["stateCounty"] = df_facts["stateCounty"].str.strip().str.lower()
    warnings.filterwarnings('ignore')

    # Get fips data for each shooting location to cross ref. demographics data
    def load_gdb_dataset(path_to_gdb):
        # Use fiona to check available layers in the GDB
        import fiona
        layers = fiona.listlayers(path_to_gdb)
        # Read the desired layer to a GeoDataFrame
        gdf = gpd.read_file(path_to_gdb, layer='ACS_2021_5YR_COUNTY')  # Change layers[0] to the layer you want to read
        return gdf

    def get_coord(row):
        lat = row["latitude"]
        long = row["longitude"]

        if lat and long:
            return Point((long, lat))
        return None

    df_main["geometry"] = df_main.apply(get_coord, axis=1)
    gdf_g = load_gdb_dataset("geodata/county/ACS_2021_5YR_COUNTY.gdb")

    test_gdf_g = gpd.GeoDataFrame(gdf_g[["GEOID", "geometry"]], geometry='geometry')
    test_df_main = gpd.GeoDataFrame(df_main, geometry='geometry')

    # Perform Spatial Join
    merged_gdf = gpd.sjoin(test_df_main, test_gdf_g, how="left", op="within")

    merged_gdf.rename(columns={'GEOID': 'fips'}, inplace=True)
    merged_gdf['fips'] = merged_gdf['fips'].astype("string")

    df_main = pd.DataFrame(merged_gdf)
    df_main = df_main.drop(columns='geometry')

    # GEOID has a 0 prefixed in case where the state id is single digit but df_facts does not have it so strip it
    df_main['fips'] = df_main['fips'].str.lstrip('0')
    df_main.sample(n=20, random_state=20)

    # Where fips is missing check by 'stateCounty'
    missing_fips_mask = df_main["fips"].isnull()
    missing_fips = df_main[missing_fips_mask].copy().reset_index()
    merged_df = missing_fips.merge(df_facts[["stateCounty", "fips"]], on='stateCounty', how='left')
    df_main.loc[missing_fips_mask, 'fips'] = merged_df.loc[missing_fips_mask, 'fips_y']

    df_main_demo = df_main.merge(df_facts[["fips", "INC110213"]], on='fips', how='left')

    # Data for two counties seems to be missing:
    # Presumably due mismatch in county boundaries during the years or new counties being formed
    missing_county_data = df_main_demo[df_main_demo['fips'].notnull() & df_main_demo['INC110213'].isnull()]

    # Household income/INC110213 is missing for some rows because states could not be identified (should be around 845)
    # in these cases just use the state average

    # Use country_facts to build a states demo. datatable
    # For states the 'state_abbreviation' field is missing and full name is used instead
    # So we need to assign correct abbr. to merge with events df.
    df_facts_state = df_facts[df_facts['state_abbreviation'].isnull()].copy()
    df_facts_state["state_abbreviation"] = df_facts_state["area_name"].map(helper.STATE_NAME_MAP)
    df_facts_state["state"] = df_facts_state["state_abbreviation"]

    df_facts_state = df_facts_state.set_index("state_abbreviation")

    df_main_demo_merged = df_main_demo.merge(df_facts_state[["state", "INC110213"]], on="state", how="left")

    df_main_demo_merged['INC110213'] = df_main_demo_merged['INC110213_x'].fillna(df_main_demo_merged['INC110213_y'])
    df_main_demo_merged.rename(columns={'INC110213_x': 'INC110213_real'}, inplace=True)
    df_main_demo_merged.drop('INC110213_y', axis=1, inplace=True)

    df_main_demo_merged["date"] = pd.to_datetime(df_main_demo_merged["date"])
    warnings.filterwarnings('default')
    
    return df_main_demo_merged, test_gdf_g

def process_main_df(df_main):
    df_main["armed_with"] = df_main["armed_with"].astype("category")

    df_main["g_race_short"] = df_main["race"].map(lambda r: "Other" if r != "White" and r != "Black" else r)

    include_types = {"shoot": "Shoot", "threat": "Weapon Visible", "point": "Pointing Weapon", "attack": "Attacked"}
    df_main["g_threat_type"] = df_main["threat_type"].map(
        lambda r: include_types[r] if r in include_types.keys() else "Other/No")


    def group_armed_with(val):
        if val == "unknown" or val == "undetermined" or val == np.NAN or val == 'nan' or val == "other":
            return "Other"
        if ";" in val:
            return random.choice(val.split(";"))

        return val

    df_main["g_armed_with"] = df_main["armed_with"].map(group_armed_with)
    
    
    return df_main

def load_state_spending_dataset():
    STATE_SPENDING_CSV = "datasets/state_spending_data/dqs_table_87_8.xls"	
    # Table 1. Real (inflation-adjusted) state and local 
    # government expenditures on police protection in the U.S., 2000-2017

    state_spending = pd.read_excel(STATE_SPENDING_CSV, sheet_name=0)
    state_spending["state_long"] = state_spending["State"]
    state_spending = state_spending.drop(columns=['State'])
    state_spending.insert(0, 'state', state_spending["state_long"].map(
        lambda x: helper.STATE_NAME_MAP[x]
    ))

    state_spending = state_spending.set_index('state')

    return state_spending

def load_homocide_df():
    # CDC homocide deaths dataset
    homocide_df = pd.read_csv("datasets/homocide_deaths/homocide_deaths.csv")
    homocide_df = homocide_df.drop(columns=["URL"])
    homocide_df = homocide_df.rename(columns={
        "STATE": "state",
        "DEATHS": "homocides"}, errors="raise")

    # Drop thousand seperators
    homocide_df["homocides"] = homocide_df["homocides"].map(lambda v: v.replace(",",""))
    homocide_df["homocides"] = homocide_df["homocides"].astype(int)

    # Get average between 2015 and 2021
    homocide_avg_df = homocide_df[homocide_df["YEAR"] == 2021]
    # homocide_avg_df = homocide_df[homocide_df["YEAR"] >= 2015]

    homocide_avg_df = homocide_avg_df.groupby(["state"]).agg(avg_homocides=('homocides', np.mean))
    
    return homocide_avg_df


def get_facts_df_readable(df):
    """
    Map each index column
    """
    _df_facts_columns = pd.read_csv("datasets/county_facts/county_facts_dictionary.csv")
    df_facts_columns = _df_facts_columns.set_index("column_name")

    return df.rename(columns=df_facts_columns.to_dict()["description"])


def get_df_facts_df():
    # Append homocide deaths per capita to demographic fats dataframe
    homocide_avg_df = load_homocide_df()
    DEMO_KEYS = {
        "All": "PST045214",
        "Black": "RHI225214",
        "White": "RHI125214",
        "Asian": "RHI425214",
        "Hispanic": "RHI725214",
        "Native American": "RHI325214",
    }

    df_facts = pd.read_csv("datasets/county_facts/county_facts.csv")

    # Load demo. facts and then group by state
    df_facts_state = df_facts[df_facts['state_abbreviation'].isnull()].copy()
    df_facts_state["state_abbreviation"] = df_facts_state["area_name"].map(helper.STATE_NAME_MAP)
    df_facts_state["state"] = df_facts_state["state_abbreviation"]

    df_facts_state = df_facts_state.set_index("state_abbreviation")
    df_facts_state = df_facts_state.merge(homocide_avg_df, 
                         left_index=True, 
                         right_index=True, 
                         how="left")
    df_facts_state["PST045214"] = df_facts_state["PST045214"].astype("int")

    df_facts_state["Homocide per 1000k"] = df_facts_state["avg_homocides"] / df_facts_state["PST045214"] *1000000
    
    available_races = ["White", "Black", "Asian", "Native American", "Hispanic"]
    main_df_races = [*available_races, "Other"]
    target_cols = ["All", *available_races]

    for c in target_cols:
        if not c in DEMO_KEYS:
            continue

        demo_key = DEMO_KEYS[c]

        if c == "All":
            population = df_facts_state[demo_key]
        else:
            all = df_facts_state[DEMO_KEYS["All"]]
            demo = df_facts_state[demo_key]
            demo = demo.astype("float") / 100

            population = all * demo

        df_facts_state[f"pop_{c}"] = population.astype(int)

    
    return df_facts_state