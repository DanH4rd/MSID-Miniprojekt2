import pandas as pd

def main():
    picked_column_names = ["Year Property Built", "Number of Bathrooms",
                    "Number of Rooms", "Number of Stories", "Number of Units", "Property Area",
                    "Lot Area", "Assessor Neighborhood"]
    column_types = {"Year Property Built": float, "Number of Bathrooms": float, "Number of Bedrooms": float,
                    "Number of Rooms": float, "Number of Stories": float, "Number of Units": float, "Property Area": float,
                    "Basement Area": float, "Lot Area": float, "Assessed Fixtures Value":int, "Assessed Improvement Value":int,
                    "Assessed Land Value":int, "Assessed Personal Property Value":int, "Assessor Neighborhood":str}

    print('Reading source file')
    dataFr = pd.read_csv("./Assessor_Historical_Secured_Property_Tax_Rolls.csv",sep=';', low_memory = False,
                         dtype = column_types, on_bad_lines='skip', thousands=',')

    # wybor 10 pierwszych wierszy dla testow
    #dataFr = dataFr.head(10)

    print('Calculating Assessed Value')
    # liczymy Assesed Value dla co majatku
    assessedValue = [row["Assessed Improvement Value"] + row["Assessed Land Value"]
                      for index, row in dataFr.iterrows()]

    print('Adding Assessed Value column')
    # Tworzymy dataset o potrzebnej postaci
    newDataFr = dataFr[picked_column_names].copy()                
    newDataFr = dataFr[picked_column_names].copy()
    newDataFr["Assessed Value"] = assessedValue

    print("Drop nan")
    newDataFr.dropna()
    
    print("Drop zeroes")
    newDataFr.loc[~(newDataFr==0).all(axis=1)]

    print('Filtering values')
    # usuwamy majatki bez jakiegos pola terenu lub liczby pokojow
    newDataFr = newDataFr[newDataFr['Assessed Value'] > 0]
    newDataFr = newDataFr[newDataFr['Property Area'] + newDataFr['Lot Area'] > 0]
    newDataFr = newDataFr[newDataFr['Number of Bathrooms']
                          + newDataFr['Number of Rooms'] + newDataFr['Number of Stories'] + newDataFr['Number of Units'] > 0]
    

    #print('Sorting values')
    #newDataFr = newDataFr.sort_values(by=['Assessed Value'], ascending=False)
    

    print(newDataFr)

 

    print('Writing to file')
    
    newDataFr.to_csv("./clean_data_without_nans_and_zeroes.csv")

    print('End of work')
 


if __name__ == "__main__":
    main()
