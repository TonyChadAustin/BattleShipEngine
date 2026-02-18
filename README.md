HOW TO USE:

If you don't wanna go through a lot of effort - the easiest way to use this program is to go to the very bottom where it says:
engine = BattleshipEngine( #More than 4 ships NOT RECCOMENDED (very slow and takes high memory)
        board_size=10,
        ships=[
            ("Carrier", 5),
            ("Battleship", 4),
            #("Cruiser", 3),
            ("LShape", [(0,0), (1,0), (0,1)]),
            #("Submarine", 3),
            #("Destroyer", 2)
        ]
    )

    here you define ships. i.e. comment out the LShape, and uncommont cruiser and submarine to generate the valid states for those 4.
    Even 4 might be a bit slow, upwards of an hour to generate the first time, and will hold onto 17-21GB of RAM/memory. 
    After generating it once the load in will be much faster (from the file it created the first time). Though it will still need the RAM to run.
    Considering the smaller ships are more "obvious" I suggest just using 3 ships for the engine and it will be practically instant!
