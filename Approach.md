# Acea Smart Water Analytics

## ML Workflow

1. Buisness Goal
    - The challenge is to determine how features influence the water availability of each presented waterbody
    - gaining a better understanding of volumes, they will be able to ensure water availability for each time interval of the year.

2. Get Data
    * Aquifer:
        - Auser:

            Description: This waterbody consists of two subsystems, called NORTH and SOUTH, where the former partly influences the behavior of the latter. Indeed, the north subsystem is a water table (or unconfined) aquifer while the south subsystem is an artesian (or confined) groundwater.

            The levels of the NORTH sector are represented by the values of the SAL, PAG, CoS and DIEC wells, while the levels of the SOUTH sector by the LT2 well.

                Features:                           *Missing values:*      **Date Range with values**      *target*

                Rainfall_Gallicano                         2859             2006-01-01 - 2020-06-30         
                Rainfall_Pontetetto                        2859             2006-01-01 - 2020-06-30 
                Rainfall_Monte_Serra                       2865             2006-01-01 - 2020-06-30 
                Rainfall_Orentano                          2859             2006-01-01 - 2020-06-30 
                Rainfall_Borgo_a_Mozzano                   2859             2006-01-01 - 2020-06-30 
                Rainfall_Piaggione                         3224             2006-01-01 - 2020-06-30 
                Rainfall_Calavorno                         2859             2006-01-01 - 2020-06-30 
                Rainfall_Croce_Arcana                      2859             2006-01-01 - 2020-06-30 
                Rainfall_Tereglio_Coreglia_Antelminelli    2859             2006-01-01 - 2020-06-30 
                Rainfall_Fabbriche_di_Vallico              2859             2006-01-01 - 2020-06-30 
                Depth_to_Groundwater_LT2                   3352             2006-01-01 - 2020-06-30         y - SOUTH x
                Depth_to_Groundwater_SAL                   3609             2007-01-05 - 2020-06-30         y - NORTH x
                Depth_to_Groundwater_PAG                   4347             2009-01-01 - 2020-06-30         y - NORTH
                Depth_to_Groundwater_CoS                   3839             2006-01-01 - 2020-06-30         y - NORTH x
                Depth_to_Groundwater_DIEC                  4884             2011-01-02 - 2020-06-30         y - NORTH
                Temperature_Orentano                          0             1998-01-04 - 2020-06-30
                Temperature_Monte_Serra                       0             1998-01-04 - 2020-06-30
                Temperature_Ponte_a_Moriano                   0             1998-01-04 - 2020-06-30
                Temperature_Lucca_Orto_Botanico               0             1998-01-04 - 2020-06-30
                Volume_POL                                 2494             2005-01-01 - 2020-06-30
                Volume_CC1                                 2494             2005-01-01 - 2020-06-30
                Volume_CC2                                 2494             2005-01-01 - 2020-06-30
                Volume_CSA                                 2494             2005-01-01 - 2020-06-30
                Volume_CSAL                                2494             2005-01-01 - 2020-06-30
                Hydrometry_Monte_S_Quirico                  913             2000-01-01 - 2020-06-30
                Hydrometry_Piaggione                       2035             2000-01-01 - 2020-06-30

                -> cut of 2006-01-01 - 2020-06-30

        - Aquifer_Doganella: