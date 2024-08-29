import geopandas as gpd
import fiona as fiona


def load_gdb_dataset(dataset_path):
    layers = fiona.listlayers(dataset_path)
    for layername in layers:
        print(layername)
        # according to docs settin ignore_geometry=True loads gpd file to pandas dataframe
        # geodata = gpd.read_file(dataset_path, driver='fileGDB', layer=layername, ignore_geometry=True)
        # geodata.to_csv(project_path + f'{layername}.csv')

    mallard_0 = gpd.read_file(dataset_path, layer=0)

    # # geopandas included map, filtered to just this hemisphere
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    westhem = world[(world['continent'] == 'North America') |
                    (world['continent'] == 'South America')]

    base = westhem.plot(color='white', edgecolor='black', figsize=(11, 11))

    mallard_0.plot(ax=base, color='red', alpha=.5)


if __name__ == '__main__':
    load_gdb_dataset("../geodata/csa/ACS_2021_5YR_CSA.gdb")